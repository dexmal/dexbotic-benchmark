"""
VLN-CE Environment Evaluator
"""
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import time
import numpy as np
import tqdm
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from evaluation.evaluator.base_evaluator import BaseEvaluator

from evaluation.utils.habitat_extensions.vlnce_tools import construct_envs_auto_reset_false
from evaluation.utils.habitat_extensions.vlnce_tools import extract_instruction_tokens
from evaluation.utils.habitat_extensions.vlnce_tools import observations_to_image, generate_video
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.common.obs_transformers import get_active_obs_transforms
from habitat_baselines.utils.common import batch_obs
from habitat import logger


class VLNCEEvaluator(BaseEvaluator):
    """
    VLN-CE Environment Evaluator
    """

    def setup_environment(self) -> Any:
        """Setup VLN-CE environment"""
        # Clone the habitat configuration from TASK_CONFIG
        self.habitat_config = self.config.TASK_CONFIG.clone()
        self.habitat_config.defrost()

        # Add RL configuration for observation transforms
        if hasattr(self.config, 'RL'):
            self.habitat_config.RL = self.config.RL.clone()

        # Add NUM_ENVIRONMENTS for environment construction
        self.habitat_config.NUM_ENVIRONMENTS = 1  # Single environment for evaluation

        # Add SIMULATOR_GPU_IDS for GPU assignment
        self.habitat_config.SIMULATOR_GPU_IDS = [0]  # Use GPU 0 for evaluation

        # Set evaluation split and dataset configuration
        split = getattr(self.config.EVAL, 'SPLIT', 'val_unseen')
        self.habitat_config.DATASET.SPLIT = split
        self.habitat_config.DATASET.ROLES = ["guide"]
        self.habitat_config.DATASET.LANGUAGES = getattr(self.config.EVAL, 'LANGUAGES', ["en-US"])

        # Set NDTW configuration and evaluation settings
        self.habitat_config.TASK.NDTW.SPLIT = split
        self.habitat_config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.habitat_config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1

        # Set chunking parameters
        self.habitat_config.DATASET.NUM_CHUNKS = getattr(self.config.TASK_CONFIG.DATASET, 'NUM_CHUNKS', 1)
        self.habitat_config.DATASET.CHUNK_IDX = getattr(self.config.TASK_CONFIG.DATASET, 'CHUNK_IDX', 0)

        # Add video measurements if needed
        if self._should_record_video():
            self.habitat_config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        # Set results directory - use output_dir if available, otherwise use RESULTS_DIR
        dataset_type = getattr(self.config.TASK_CONFIG.DATASET, 'TYPE', 'r2r')
        
        # Prefer output_dir from config (dexbotic-benchmark style) over RESULTS_DIR (habitat style)
        if hasattr(self.config, 'output_dir') and self.config.output_dir:
            # Use output_dir as base, but still organize by dataset type and split
            base_output_dir = self.config.output_dir
            results_dir = os.path.join(base_output_dir, dataset_type, split)
        else:
            # Fallback to habitat-style RESULTS_DIR
            results_dir = os.path.join(
                self.config.RESULTS_DIR,
                "vlnce",
                dataset_type,
                split
            )
        os.makedirs(results_dir, exist_ok=True)

        # Update config with new results directory
        self.config = self.config.clone()
        self.config.defrost()
        self.config.RESULTS_DIR = results_dir
        # Use output_structure's videos_dir if available, otherwise use results_dir/videos
        if hasattr(self, 'output_structure') and 'videos_dir' in self.output_structure:
            self.config.VIDEO_DIR = str(self.output_structure['videos_dir'])
        else:
            self.config.VIDEO_DIR = os.path.join(results_dir, "videos")
        self.config.freeze()

        # Setup observation transforms
        self.obs_transforms = get_active_obs_transforms(self.habitat_config)

        # Create a config object with TASK_CONFIG for construct_envs
        env_config = self.config.clone()
        env_config.defrost()
        env_config.TASK_CONFIG = self.habitat_config
        env_config.freeze()

        # Create environment
        env = construct_envs_auto_reset_false(env_config, get_env_class(env_config.ENV_NAME))  # env_class will be determined by habitat

        return env

    def setup_model(self) -> Any:
        """Setup VLN-CE VLA agent"""
        try:
            # Import VLN-CE VLA agent
            from evaluation.policies.vlnce_vla_agent import VLNCEVLAAgent

            logger.info("Loading VLN-CE VLA agent")
            model = VLNCEVLAAgent(self.config)

            return model

        except ImportError as e:
            logger.error(f"Unable to import VLN-CE VLA agent: {e}")
            raise

    def _run_evaluation_impl(self) -> Dict[str, Any]:
        """Implement VLN-CE evaluation logic"""
        try:
            # Get evaluation configuration
            eval_config = self.config.EVAL if hasattr(self.config, 'EVAL') else type('obj', (object,), {})()
            split = self.habitat_config.DATASET.SPLIT
            save_results = eval_config.get("SAVE_RESULTS", True)
            episode_count = eval_config.get("EPISODE_COUNT", -1)

            logger.info(f"Starting VLN-CE evaluation, split: {split}")

            # Setup results file
            results_fname = None
            if save_results:
                results_fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"{split}_{self.habitat_config.DATASET.NUM_CHUNKS}-{self.habitat_config.DATASET.CHUNK_IDX}.json",
                )

            # Reset environment
            observations = self.env.reset()
            observations = extract_instruction_tokens(observations, self.habitat_config.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, device="cpu")  # Use CPU for observations
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            stats_episodes = {}

            # Initialize tracking variables for video recording
            rgb_frames = [[] for _ in range(self.env.num_envs)]

            # Setup video recording
            if self._should_record_video():
                os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

            num_eps = sum(self.env.number_of_episodes)
            if episode_count > -1:
                num_eps = min(episode_count, num_eps)

            # Setup progress bar
            use_pbar = getattr(self.config, 'use_pbar', True)
            pbar = tqdm.tqdm(total=num_eps) if use_pbar else None

            log_str = (
                f"[VLN-CE Evaluation] [Episodes evaluated: {0}/{num_eps}] "
                "[Time elapsed (s): {time}]"
            )
            start_time = time.time()

            assert self.env.num_envs == 1, "Only support num_envs=1 for VLN-CE evaluation"

            episode_first_frame = True

            # Main evaluation loop
            while self.env.num_envs > 0 and len(stats_episodes) < num_eps:
                current_episodes = self.env.current_episodes()

                # Get current observation
                curr_rgb = batch[0]["rgb"].cpu().numpy() if "rgb" in batch[0] else batch[0]["rgb_0"].cpu().numpy()

                # Get instruction
                instruction = current_episodes[0].instruction.instruction_text

                # Get action from VLA agent
                action = self.model.step(
                    obs={"rgb": curr_rgb},
                    goal=instruction,
                    episode_first_frame=episode_first_frame
                )

                # Execute action
                outputs = self.env.step([action])

                observations, _, dones, infos = [list(x) for x in zip(*outputs)]

                # Handle episode completion
                for i in range(self.env.num_envs):
                    # Handle video recording
                    if self._should_record_video():
                        frame = observations_to_image(observations[i], infos[i])
                        frame = self._add_instruction_to_image(frame, current_episodes[i].instruction.instruction_text)
                        rgb_frames[i].append(frame)

                    if not dones[i]:
                        episode_first_frame = False
                        continue

                    # Episode ended
                    ep_id = current_episodes[i].episode_id
                    stats_episodes[ep_id] = infos[i]

                    # Reset for next episode
                    observations[i] = self.env.reset_at(i)[0]
                    episode_first_frame = True
                    # Reset VLA agent to clear action queue from previous episode
                    # This ensures that when text changes (new instruction), 
                    # the agent will immediately call VLA service with episode_first_frame=True
                    self.model.reset()

                    if use_pbar:
                        pbar.update()
                    else:
                        logger.info(
                            log_str.format(
                                evaluated=len(stats_episodes),
                                total=num_eps,
                                time=round(time.time() - start_time),
                            )
                        )

                    # Generate video if enabled
                    if self._should_record_video():
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=ep_id,
                            checkpoint_idx="0",
                            metrics={"spl": stats_episodes[ep_id].get("spl", 0.0)},
                            tb_writer=None,
                        )
                        # Remove top_down_map to save space
                        if "top_down_map_vlnce" in stats_episodes[ep_id]:
                            del stats_episodes[ep_id]["top_down_map_vlnce"]
                        rgb_frames[i] = []

                # Rebuild observations for the next step using the latest data (after potential resets)
                observations = extract_instruction_tokens(
                    observations,
                    self.habitat_config.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, device="cpu")
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                # Pause environments for completed episodes
                envs_to_pause = []
                next_episodes = self.env.current_episodes()

                for i in range(self.env.num_envs):
                    if next_episodes[i].episode_id in stats_episodes:
                        envs_to_pause.append(i)

                (self.env, batch, rgb_frames,) = self._pause_envs(
                    envs_to_pause,
                    self.env,
                    batch,
                    rgb_frames,
                )

            # Cleanup
            self.env.close()
            if use_pbar:
                pbar.close()

            # Save results
            if save_results and results_fname:
                with open(results_fname, "w") as f:
                    json.dump(stats_episodes, f, indent=4)

            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(stats_episodes)

            return summary_stats

        except Exception as e:
            logger.error(f"Error occurred during VLN-CE evaluation: {e}")
            return {
                "error": str(e),
                "total_episodes": 0,
                "successful_episodes": 0,
                "average_spl": 0.0,
                "average_success": 0.0,
            }

    def _should_record_video(self) -> bool:
        """Check if video recording is enabled"""
        return hasattr(self.config, 'VIDEO_OPTION') and len(getattr(self.config, 'VIDEO_OPTION', [])) > 0

    def _add_instruction_to_image(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """Append instruction text underneath the image for visualization.
        
        The returned image has white text on a black background. Uses textwrap to
        split long text into multiple lines.
        
        Args:
            image: the image to put text underneath
            instruction: a string instruction to display
            
        Returns:
            A new image with text inserted underneath the input image
        """
        import cv2
        import textwrap

        h, w, c = image.shape
        font_size = 0.5
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Create blank image for text (same width as input image)
        blank_image = np.zeros(image.shape, dtype=np.uint8)
        
        # Calculate character size for text wrapping
        char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
        wrapped_text = textwrap.wrap(instruction, width=int(w / char_size[0]))

        # Draw text on blank image
        y = 0
        for line in wrapped_text:
            textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
            y += textsize[1] + 10  # Add line height plus spacing
            x = 10
            cv2.putText(
                blank_image,
                line,
                (x, y),
                font,
                font_size,
                (255, 255, 255),  # White text
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        # Crop blank image to actual text height
        text_image = blank_image[0 : y + 10, 0:w]
        
        # Concatenate original image with text image
        final = np.concatenate((image, text_image), axis=0)
        
        return final

    @staticmethod
    def is_valid_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and np.isfinite(value)

    def _calculate_summary_stats(self, stats_episodes: Dict) -> Dict[str, Any]:
        """Calculate summary statistics from episode results"""
        distance_to_goal_values = []
        success_values = []
        spl_values = []
        ndtw_values = []
        path_lengths = []
        oracle_success_values = []
        steps_taken_values = []

        invalid_spls = 0
        invalid_distances = 0

        for ep_stats in stats_episodes.values():
            if "spl" in ep_stats:
                if self.is_valid_number(ep_stats["spl"]):
                    spl_values.append(ep_stats["spl"])
                else:
                    invalid_spls += 1
            if "distance_to_goal" in ep_stats:
                if self.is_valid_number(ep_stats["distance_to_goal"]):
                    distance_to_goal_values.append(ep_stats["distance_to_goal"])
                else:
                    invalid_distances += 1
            if "success" in ep_stats:
                success_values.append(ep_stats["success"])
            if "path_length" in ep_stats:
                path_lengths.append(ep_stats["path_length"])
            if "ndtw" in ep_stats:
                ndtw_values.append(ep_stats["ndtw"])
            if "oracle_success" in ep_stats:
                oracle_success_values.append(ep_stats["oracle_success"])
            if "steps_taken" in ep_stats:
                steps_taken_values.append(ep_stats["steps_taken"])

        return {
            "total_episodes": len(stats_episodes),
            "successful_episodes": sum(success_values),
            "average_spl": float(np.mean(spl_values)) if spl_values else 0.0,
            "average_success": float(np.mean(success_values)) if success_values else 0.0,
            "average_path_length": float(np.mean(path_lengths)) if path_lengths else 0.0,
            "average_distance_to_goal": float(np.mean(distance_to_goal_values)) if distance_to_goal_values else 0.0,
            "average_ndtw": float(np.mean(ndtw_values)) if ndtw_values else 0.0,
            "average_oracle_success": float(np.mean(oracle_success_values)) if oracle_success_values else 0.0,
            "average_steps_taken": float(np.mean(steps_taken_values)) if steps_taken_values else 0.0,
            "invalid_spls": invalid_spls,
            "invalid_distances": invalid_distances,
        }

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        batch,
        rgb_frames=None,
    ):
        """Pause environments that have completed their episodes"""
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # Update batch
            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return envs, batch, rgb_frames
