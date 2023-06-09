import mujoco
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from enum import Enum
from typing import Optional

# import gymnasium as gym
# import gym
# from gymnasium import spaces
import gym
from gym import spaces
from gym import Env
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

import torch
import random
import os

from envs.mujoco_utils import MujocoModelNames, robot_get_obs
from misc.utils import seed_everything
from envs.env_tasks import PandaTasks



DEFAULT_CAMERA_CONFIG = {
    "distance": 2.2,
    "azimuth": 45.0,
    "elevation": -30.0,
    "lookat": np.array([0, 0, 1]),
}


class MujocoPandaEnv(gym.Env):
    """
    This enviorment is based on uses assets from the FrankaKitchenEnv from the Farama foundation: 
    https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/
    """

    def __init__(
            self,
            xml_path: str = "panda_scene.xml",
            task: PandaTasks = PandaTasks.reach,
            render_mode: str = "human",
            frame_skip: int = 10,
            obs_noise_ratio: float = 0.0,
            sb_mode: bool = False,
            episode_length: int = 100,
            random_init_qpos: bool = True,
            verbose: bool = False,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.mujoco_renderer = MujocoRenderer(
            self.model, self.data, DEFAULT_CAMERA_CONFIG
        )
        self.verbose = verbose

        self.camera = 0
        self.model_names = MujocoModelNames(self.model)

        self.frame_skip = frame_skip
        self.robot_noise_ratio = obs_noise_ratio
        self.render_mode = render_mode
        self.sb_mode = sb_mode

        # reset data
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.init_ctrl = self.data.ctrl.ravel().copy()

        # Actuator ranges
        ctrlrange = self.model.actuator_ctrlrange
        self.actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        self.actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0

        self.render_fig, self.render_ax = plt.subplots(1, 1)

        self.episode_length = episode_length
        self.episode_step_counter = 0
        self.task = task
        self.random_init_qpos = random_init_qpos
        self.episode_counter = 0

    def __repr__(self):
        return f"PandaEnv_{self.task.value}"

    @property
    def observation_space(self):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18, ),
            dtype=np.float32,)

    @property
    def action_space(self):
        return spaces.Box(
            low=-1,
            high=1,
            shape=(8,),
            dtype=np.float32)

    def render(self, save_path=None, **kwargs):
        if self.render_mode == "":
            return

        self.mujoco_renderer.render(
            self.render_mode, camera_id=self.camera
        )

    def _check_done(self):
        if self.task == PandaTasks.avoid:
            return False, ""
        
        elif self.task == PandaTasks.reach:
            # return False, ""  # I think this is better for re-using the transitions for the avoid task?

            target_sphere_xpos = self.data.site("target_sphere").xpos
            ee_xpos = self.data.site("EEF").xpos
            dist = abs(target_sphere_xpos - ee_xpos)
            if dist.mean() < 0.03:
                if self.verbose:
                    print(f"Step {self.episode_step_counter}: Reached EE target!")
                return True, "reached"
            else:
                return False, ""
            
    def compute_reward(self):
        reward_dict = {}

        # sphere distance reward
        target_sphere_xpos = self.data.site("target_sphere").xpos
        ee_xpos = self.data.site("EEF").xpos
        dist = abs(target_sphere_xpos - ee_xpos)
        # reward_dict["Reach_reward"] = -1 - 10 * dist.mean()
        #  reward_dict["Reach_reward"] = -1 - 1 * dist.mean()
        reward_dict["Reach_reward"] = -1

        # floor collision reward (eventually never used this)
        reward_dict["Floor_reward"] = 0
        for coni in range(self.data.ncon):
            con = self.data.contact[coni]
            if con.geom1 == self.data.geom("floor").id or con.geom2 == self.data.geom("floor").id:
                # print("contact with floor")
                # print('    geom1    = %d' % (con.geom1,))
                # print('    geom2    = %d' % (con.geom2,))
                reward_dict["Floor_reward"] -= 1

        # avoidance reward
        # since mujoco does not seem to give me that data, we check by hand whether any of robot geoms is in the avoid area
        # shitty code that does not generalize, but works for now
        reward_dict["Avoid_reward"] = 0
        for idx, pos in enumerate(self.data.xpos[3:-3, :]):
            if 0.1 <= pos[0] <= 1 and 0.1 <= pos[1] <= 1 and 0.01 <= pos[2] <= 10:
                if self.verbose:
                    print(f"Step {self.episode_step_counter}: Some robot link is in avoidance zone with pos {pos}!")
                # reward_dict["Avoid_reward"] -= 10
                reward_dict["Avoid_reward"] = -10  # just 10...
                break

        return reward_dict

    def step(self, action):

        # Denormalize the input action from [-1, 1] range to the each actuators control range
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = self.actuation_center + action * self.actuation_range

        # apply action and simulate for n steps
        mujoco.mj_step(self.model, self.data, self.frame_skip)

        self.render()

        obs = self._get_obs()
        trunc = self.episode_step_counter >= self.episode_length 
        self.episode_step_counter += 1
        done, done_reason = self._check_done()
        info = {
            "pos": obs,
            "done_reason": done_reason if not trunc else "truncated",
        }
        reward_dict = self.compute_reward()
        reward = reward_dict[f"{self.task.value}_reward"]
        if self.task == PandaTasks.reach:
            if done:
                reward = 100

        if done or trunc:
            self.episode_counter += 1

        info.update(reward_dict)  # merge dicts and save all task rewards

        if self.sb_mode:
            return obs, reward, done or trunc, info
        else:
            return obs, reward, done, trunc, info

    def _get_obs(self):
        # Gather simulated observation
        robot_qpos, robot_qvel = robot_get_obs(
            self.model, self.data, self.model_names.joint_names
        )

        # Simulate observation noise
        robot_qpos += self.robot_noise_ratio * np.random.uniform(
            low=-1.0, high=1.0, size=robot_qpos.shape
        )
        robot_qvel += self.robot_noise_ratio * np.random.uniform(
            low=-1.0, high=1.0, size=robot_qvel.shape
        )

        return np.concatenate((robot_qpos.copy(), robot_qvel.copy()))

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
            **kwargs
    ):
        if seed is not None:
            self.seed(seed)  # this one is important!
            try:
                super().reset(seed=seed)
            except TypeError:
                pass

        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)  # deterministic reset to initial robot configuration
        self.data.ctrl[:] = self.init_ctrl
        if self.random_init_qpos and self.episode_counter % 2 == 0:
            # to boost exploration, we randomize some of the initial joint positions
            self.data.qpos[0] = self.actuation_center[0] + np.random.uniform(-1, 1) * self.actuation_range[0]  # randomize frist two join positions...
            self.data.qpos[1] = self.actuation_center[1] + np.random.uniform(-1, 1) * self.actuation_range[1]
        else:
            self.data.qpos[0] = -1.75  # such that panda arm starts at the right of obstacle
            self.data.qpos[1] = 1.25

        mujoco.mj_step(self.model, self.data, 1)
        self.render()

        self.episode_step_counter = 0

        if self.sb_mode:
            return self._get_obs()
        else:
            return self._get_obs(), "info"


if __name__ == "__main__":
    env = MujocoPandaEnv(render_mode="human", frame_skip=10, random_init_qpos=True)

    for e in range(10):
        obs, _ = env.reset()
        print(obs)
        for _ in range(30):
            # env.step(np.random.rand(8))
            action = np.zeros(8)
            action[0] = -1
            new_obs, reward, done, tunc, info = env.step(action)
