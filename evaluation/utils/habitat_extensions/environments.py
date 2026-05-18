from typing import Any, Dict, Optional, Tuple, Union

import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry

@baseline_registry.register_env(name="VLNCECollectorEnv")
class VLNCECollectorEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._success_measure_name = "oracle_success"
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self) -> Tuple[float, float]:
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def _episode_success(self) -> bool:
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations: Observations) -> bool:
        # print(self._episode_success())
        # if self._episode_success():
        #     print("Early stop because of oracle stop.")
        return self._env.episode_over  # or self._episode_success()

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        curr_metrics = self.habitat_env.get_metrics()
        curr_metrics["early_stop"] = self._episode_success()
        return curr_metrics


@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self) -> Tuple[float, float]:
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations: Observations):
        return 0.0

    def get_done(self, observations: Observations):
        return self._env.episode_over

    def get_info(self, observations: Observations):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(agent_state.rotation.inverse(), np.array([0, 0, -1]))
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }


@baseline_registry.register_env(name="VLNCEWaypointEnv")
class VLNCEWaypointEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None) -> None:
        self._rl_config = config.RL
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self) -> Tuple[float, float]:
        return (
            np.finfo(np.float).min,
            np.finfo(np.float).max,
        )

    def get_reward(self, observations: Observations) -> float:
        return self._env.get_metrics()[self._reward_measure_name]

    def _episode_success(self) -> bool:
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over or self._episode_success()

    def get_info(self, observations: Observations) -> Dict[str, Any]:
        return self.habitat_env.get_metrics()

    def get_num_episodes(self) -> int:
        return len(self.episodes)

