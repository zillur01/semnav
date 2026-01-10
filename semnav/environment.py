from typing import Optional

import habitat
import numpy as np
from habitat import Config, Dataset


@habitat.registry.register_env(name="SimpleRLEnv")
class SimpleRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)

    def reset(self):
        observations = super().reset()
        return observations

    def step(self, *args, **kwargs):
        step_result = super().step(*args, **kwargs)
        step_result[3]['states'] = self.get_sensor_pose()

        return step_result

    def get_reward_range(self):
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        return self._env.get_metrics()[self.config.TASK.SUCCESS_MEASURE]

    def get_done(self, observations):
        if self._env.episode_over:
            return True
        if self._env.get_metrics()[self.config.TASK.SUCCESS_MEASURE]:
            return True
        return False

    def get_info(self, observations):
        return self._env.get_metrics()
    
    def get_sensor_pose(self):
        return self._env.sim.get_agent_state()
    
    def get_annotations(self):
        regions = self._env._sim.semantic_annotations().regions  # type: ignore
    
    def get_camera_intrinsics(self, sensor_name):
        # Get render camera
        render_camera = self._env._sim._sensors[sensor_name]._sensor_object.render_camera

        # Get projection matrix
        projection_matrix = render_camera.projection_matrix

        # Get resolution
        viewport_size = render_camera.viewport

        # Intrinsic calculation
        fx = projection_matrix[0, 0] * viewport_size[0] / 2.0
        fy = projection_matrix[1, 1] * viewport_size[1] / 2.0
        cx = (projection_matrix[2, 0] + 1.0) * viewport_size[0] / 2.0
        cy = (projection_matrix[2, 1] + 1.0) * viewport_size[1] / 2.0

        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])
        return intrinsics


# @habitat.registry.register_env(name="SimpleRLEnv")
# class NavRLEnv(habitat.RLEnv):
#     def __init__(self, config: Config, dataset: Optional[Dataset] = None):
#         super().__init__(config, dataset)
#         self._reward_measure_name = self.config.TASK.REWARD_MEASURE
#         self._success_measure_name = self.config.TASK.SUCCESS_MEASURE

#         self._previous_measure: Optional[float] = None

#     def reset(self):
#         observations = super().reset()
#         self._previous_measure = self._env.get_metrics()[
#             self._reward_measure_name
#         ]
#         return observations

#     def step(self, *args, **kwargs):
#         return super().step(*args, **kwargs)

#     def get_reward_range(self):
#         return (
#             self.config.TASK.SLACK_REWARD - 1.0,
#             self.config.TASK.SUCCESS_REWARD + 1.0,
#         )

#     def get_reward(self, observations):
#         reward = self.config.TASK.SLACK_REWARD

#         current_measure = self._env.get_metrics()[self._reward_measure_name]

#         reward += self._previous_measure - current_measure
#         self._previous_measure = current_measure

#         if self._episode_success():
#             reward += self.config.TASK.SUCCESS_REWARD

#         return reward

#     def _episode_success(self):
#         return self._env.get_metrics()[self._success_measure_name]

#     def get_done(self, observations):
#         done = False
#         if self._env.episode_over or self._episode_success():
#             done = True
#         return done

#     def get_info(self, observations):
#         return self.habitat_env.get_metrics()
