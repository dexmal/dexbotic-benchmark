import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import random
import numpy as np
from numpy import ndarray
from typing import List, Optional, Type, Union, Dict, Any
from evaluation.utils.habitat_extensions import maps
import habitat
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
from habitat.core.dataset import ALL_SCENES_MASK
from habitat_baselines.utils.env_utils import make_env_fn
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps as habitat_maps
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

cv2 = try_cv2_import()

def construct_envs(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
    auto_reset_done: bool = True,
    episodes_allowed: Optional[List[str]] = None,
) -> VectorEnv:
    """Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param auto_reset_done: Whether or not to automatically reset the env on done

    :return: VectorEnv object created according to specification.
    """

    num_envs_per_gpu = config.NUM_ENVIRONMENTS
    if isinstance(config.SIMULATOR_GPU_IDS, list):
        gpus = config.SIMULATOR_GPU_IDS
    else:
        gpus = [config.SIMULATOR_GPU_IDS]
    num_gpus = len(gpus)
    num_envs = num_gpus * num_envs_per_gpu

    if episodes_allowed is not None:
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episodes_allowed
        config.freeze()

    configs = []
    env_classes = [env_class for _ in range(num_envs)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if ALL_SCENES_MASK in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multi-process logic relies on being able"
                " to split scenes uniquely between processes"
            )

        if len(scenes) < num_envs and len(scenes) != 1:
            raise RuntimeError("reduce the number of GPUs or envs as there" " aren't enough number of scenes")

        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)

        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_gpus):
        for j in range(num_envs_per_gpu):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_envs_per_gpu) + j

            task_config = proc_config.TASK_CONFIG
            task_config.SEED += proc_id
            if len(scenes) > 0:
                task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]

            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]

            task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

            proc_config.freeze()
            configs.append(proc_config)

    envs = habitat.ThreadedVectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)),
        auto_reset_done=auto_reset_done,
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs


def construct_envs_auto_reset_false(config: Config, env_class: Type[Union[Env, RLEnv]]) -> VectorEnv:
    return construct_envs(config, env_class, auto_reset_done=False)


def extract_instruction_tokens(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    tokens_uuid: str = "tokens",
) -> Dict[str, Any]:
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure.
    """
    if instruction_sensor_uuid not in observations[0] or instruction_sensor_uuid == "pointgoal_with_gps_compass":
        return observations
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            observations[i][instruction_sensor_uuid] = observations[i][instruction_sensor_uuid]["tokens"]
        else:
            break
    return observations


def observations_to_image(observation: Dict[str, Any], info: Dict[str, Any]) -> ndarray:
    """Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    if "rgb" in observation and len(observation["rgb"].shape) == 4:
        return pano_observations_to_image(observation, info)
    elif "depth" in observation and len(observation["depth"].shape) == 4:
        return pano_observations_to_image(observation, info)

    egocentric_view = []
    observation_size = -1
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb)

    # draw depth map if observation has depth info. resize to rgb size.
    if "depth" in observation:
        if observation_size == -1:
            observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        depth_map = cv2.resize(
            depth_map,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(depth_map)

    assert len(egocentric_view) > 0, "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    frame = egocentric_view

    map_k = None
    if "top_down_map_vlnce" in info:
        map_k = "top_down_map_vlnce"
    elif "top_down_map" in info:
        map_k = "top_down_map"

    if map_k is not None:
        td_map = info[map_k]["map"]

        td_map = maps.colorize_topdown_map(
            td_map,
            info[map_k]["fog_of_war_mask"],
            fog_of_war_desat_amount=0.75,
        )
        td_map = habitat_maps.draw_agent(
            image=td_map,
            agent_center_coord=info[map_k]["agent_map_coord"],
            agent_rotation=info[map_k]["agent_angle"],
            agent_radius_px=min(td_map.shape[0:2]) // 24,
        )
        if td_map.shape[1] < td_map.shape[0]:
            td_map = np.rot90(td_map, 1)

        if td_map.shape[0] > td_map.shape[1]:
            td_map = np.rot90(td_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = td_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        td_map = cv2.resize(
            td_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((egocentric_view, td_map), axis=1)
    return frame


def pano_observations_to_image(observation: Dict[str, Any], info: Dict[str, Any]) -> ndarray:
    """Creates a rudimentary frame for a panoramic observation. Includes RGB,
    depth, and a top-down map.
    TODO: create a visually-pleasing stitched panorama frame
    """
    pano_frame = []
    channels = None
    rgb = None
    if "rgb" in observation:
        cnt = observation["rgb"].shape[0]
        rgb = observation["rgb"][[*range(cnt // 2, cnt), *range(cnt // 2)], :, :, :]
        channels = rgb.shape[3]
        vert_bar = np.ones((rgb.shape[1], 20, channels)) * 255
        rgb_frame = [rgb[0]]
        for i in range(1, rgb.shape[0]):
            rgb_frame.append(vert_bar)
            rgb_frame.append(rgb[i])
        pano_frame.append(np.concatenate(rgb_frame, axis=1))

    if "depth" in observation:
        cnt = observation["depth"].shape[0]
        observation["depth"] = observation["depth"][[*range(cnt // 2, cnt), *range(cnt // 2)], :, :, :]
        if len(pano_frame) > 0:
            assert observation["depth"].shape[0] == rgb.shape[0]
            pano_frame.append(np.ones((20, pano_frame[0].shape[1], channels)) * 255)
            observation_size = rgb.shape[1:3]
        else:
            observation_size = observation["depth"].shape[1:3]

        vert_bar = np.ones((observation_size[0], 20, 3)) * 255

        depth = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth = np.stack([depth for _ in range(3)], axis=3)

        depth_frame = [cv2.resize(depth[0], dsize=observation_size, interpolation=cv2.INTER_CUBIC)]
        for i in range(1, depth.shape[0]):
            depth_frame.append(vert_bar)
            depth_frame.append(
                cv2.resize(
                    depth[i],
                    dsize=observation_size,
                    interpolation=cv2.INTER_CUBIC,
                )
            )
        pano_frame.append(np.concatenate(depth_frame, axis=1))

    pano_frame = np.concatenate(pano_frame, axis=0)

    if "top_down_map_vlnce" in info:
        k = "top_down_map_vlnce"
    elif "top_down_map" in info:
        k = "top_down_map"
    else:
        k = None

    if k is not None:
        top_down_map = info[k]["map"]
        top_down_map = maps.colorize_topdown_map(top_down_map, info[k]["fog_of_war_mask"])
        map_agent_pos = info[k]["agent_map_coord"]
        top_down_map = habitat_maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=info[k]["agent_angle"],
            agent_radius_px=min(top_down_map.shape[0:2]) // 24,
        )
        if top_down_map.shape[1] < top_down_map.shape[0]:
            top_down_map = np.rot90(top_down_map, 1)

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = top_down_map.shape
        top_down_width = pano_frame.shape[1] // 3
        top_down_height = int(top_down_width / old_w * old_h)

        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        white = np.ones((top_down_height, pano_frame.shape[1] - top_down_width, 3)) * 255
        top_down_map = np.concatenate((white, top_down_map), axis=1)
        pano_frame = np.concatenate((pano_frame, top_down_map), axis=0)

    return pano_frame.astype(np.uint8)

def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[ndarray],
    episode_id: Union[str, int],
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
) -> None:
    """Generate video according to specified information. Using a custom
    verion instead of Habitat's that passes FPS to video maker.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(metric_strs)
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name, fps=fps)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(f"episode{episode_id}", checkpoint_idx, images, fps=fps)
