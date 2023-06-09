from gym.utils import EzPickle
from gym import spaces
from gym import Env

import math
import numpy as np
import matplotlib.pyplot as plt
from misc.utils import se_kernel, seed_everything
from envs.env_tasks import PointTasks


class PointNavEnv(Env, EzPickle):
    """

    State: position.
    Action: velocity.
    """

    def __init__(self,
                 render_mode=None,
                 sb_mode=False,
                 task=PointTasks.obstacle,
                 detect_collisions=False,
                 ):

        super().__init__()
        EzPickle.__init__(**locals())

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.task = task
        self.detect_collisions = detect_collisions

        self.xlim = (-10, 10)
        self.ylim = (-10, 10)
        self.obstacle_patches = [
            Rectangle(-5, 3, 10, 2),  # top vertical rectangle of inverted U
            Rectangle(-5, -5, 2, 8),  # the left horizontal rectangle of the inverted U
            Rectangle(3, -5, 2, 8)  # the right horizontal rectangle of the inverted U
        ]

        self.vel_bound = 0.8
        self.observation = None
        self.sb_mode = sb_mode
        self.reset()

        self._ax = None
        self._env_lines = []
        self.fixed_plots = None
        self.dynamic_plots = []

        self.step_counter = 0
        self.done_reason = ""

        self.render_mode = render_mode
        self.show_obstacles = True if self.task == PointTasks.obstacle else False
        if render_mode == "human":
            self.render_fig, self.render_ax = plt.subplots()
            self.render_fig.show()

    def __repr__(self):
        return f"PointNavEnv_{self.task.value}"

    @property
    def observation_space(self):
        return spaces.Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            dtype=np.float32,
            shape=None)

    @property
    def action_space(self):
        return spaces.Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim,),
            dtype=np.float32)

    def get_current_obs(self):
        return np.copy(self.observation)

    def reset(self, init_state=None, seed=None, **kwargs):
        if seed is not None:
            self.seed(seed)  # this one is important!
            try:
                super().reset(seed=seed)
            except TypeError:
                pass

        if init_state is not None:
            assert not self._check_done(init_state)
        else:
            done = True
            while done:
                # unclipped_observation = np.random.uniform(*tuple(self.xlim), size=2)
                init_state = np.random.normal(loc=0, scale=3, size=2)  # PPO learns better inside obstacle with this...
                # init_state = np.random.uniform(-10, 10, size=2)
                init_state = np.clip(
                    init_state,
                    self.observation_space.low,
                    self.observation_space.high)
                done = self._check_done(init_state)

        self.step_counter = 0
        self.done_reason = ""

        self.observation = init_state

        if self.sb_mode:
            return self.observation
        else:
            return self.observation, "info"

    def _check_done(self, obs):
        # different tasks have different ermination conditions
        # if self.task == PointTasks.obstacle or self.task == PointTasks.obstacle_top or self.detect_collisions:
        if self.task == PointTasks.obstacle or self.detect_collisions:
            for rect in self.obstacle_patches:
                dist = rect.calc_distance_to_point(*tuple(obs))
                if dist <= 0:
                    self.done_reason = "obstacle collision"
                    return True

        # the reach tasks never terminate early
        elif self.task == PointTasks.top_reach:
            return False

        elif self.task == PointTasks.side_reach:
            return False

        elif self.task == PointTasks.obstacle_top:
            return False

        else:
            raise NotImplementedError

        return False

    def compute_reward(self, observation, action, task=None):
        # different tasks have different reward functions
        if task is None:
            task = self.task

        if task == PointTasks.obstacle or task == PointTasks.obstacle_top:
            # penalize squared dist to closest obstacle
            distances = []
            for rect in self.obstacle_patches:
                distances.append(rect.calc_distance_to_point(*tuple(observation)))
            closest_obstacle_dist = np.min(distances)
            obstacle_punish = se_kernel(closest_obstacle_dist, 0)

            reward = -1 * obstacle_punish
            if closest_obstacle_dist <= 0:
                    reward -= 10
                    # reward -= 20

            if task == PointTasks.obstacle_top:
                if observation[1] < 7:
                    reward -= 5
                    # reward -= 1

        # reach tasks have a simple reward function
        elif task == PointTasks.top_reach:
            if observation[1] < 7:
                # reward = -5
                reward = -1
            else:
                reward = 0

        elif task == PointTasks.side_reach:
            if observation[0] < 7:
                reward = -1
            else:
                reward = 0

        else:
            raise NotImplementedError

        return reward

    def step(self, action):
        action = action.ravel()

        action_magnitude = np.linalg.norm(action)
        if action_magnitude > 1:
            # make action a unit vector, this punishes the agent for useless magnitude on x
            action = action / action_magnitude

        observation = self.dynamics.forward(self.observation, action)
        observation = np.clip(
            observation,
            self.observation_space.low,
            self.observation_space.high)

        reward = self.compute_reward(observation, action)
        done = self._check_done(observation)

        obstacle_reward = self.compute_reward(observation, action, task=PointTasks.obstacle)
        top_reward = self.compute_reward(observation, action, task=PointTasks.top_reach)
        side_reward = self.compute_reward(observation, action, task=PointTasks.side_reach)
        obst_top_reward = self.compute_reward(observation, action, task=PointTasks.obstacle_top)

        self.observation = np.copy(observation)
        trunc = self.step_counter >= 100

        self.step_counter += 1

        info_dict = {
            'pos': observation,
            'done_reason': self.done_reason,
            'obstacle_reward': obstacle_reward,
            'top_reward': top_reward,
            'side_reward': side_reward,
            'obst_top_reward': obst_top_reward
        }

        if self.sb_mode:
            return observation, reward, done or trunc, info_dict
        else:
            return observation, reward, done, trunc, info_dict

    def render(self, ignore_render_mode=False, save_path="", show_obstacle=True, *args, **kwargs):
        if self.render_mode != "human" and not ignore_render_mode:
            return

        self.render_fig.gca().cla()

        # plot obstacle rectangles
        if show_obstacle:
            for rect in self.obstacle_patches:
                rectangle = plt.Rectangle((rect.x_start, rect.y_start), rect.width, rect.height, fc='black', ec="black")
                self.render_ax.add_patch(rectangle)

        self.render_ax.scatter(self.observation[0], self.observation[1], s=400, c="red")

        self.render_ax.set_xlim(self.xlim[0], self.xlim[1])
        self.render_ax.set_ylim(self.ylim[0], self.ylim[1])
        self.render_ax.set_title(f"Env state {self.observation}")

        self.render_fig.canvas.draw()
        self.render_fig.canvas.flush_events()

        if save_path:
            self.render_fig.savefig(save_path)

    def plot_obstacles(self, fig):
        # plot obstacle rectangles
        for rect in self.obstacle_patches:
            rectangle = plt.Rectangle((rect.x_start, rect.y_start), rect.width, rect.height, fc='black', ec="black")
            fig.gca().add_patch(rectangle)

    def plot_reward(self, action=np.array([0, 0])):
        f = plt.figure(figsize=(8, 8))

        x_min, x_max = tuple(np.array(self.xlim))
        y_min, y_max = tuple(np.array(self.ylim))

        img = np.zeros((100, 100))
        for x_idx, x in enumerate(np.linspace(x_max, x_min, 100)):
            for y_idx, y in enumerate(np.linspace(y_max, y_min, 100)):
                r = self.compute_reward(np.array([x, y]), action)
                img[y_idx, x_idx] = r

        # plot the goal bar at the top
        rectangle = plt.Rectangle((0, 9), 10, 1, fc='black', ec="black")
        self.render_ax.add_patch(rectangle)

        plt.imshow(img)
        cbar = plt.colorbar()
        cbar.set_label("Reward")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def plot_initial_dist(self, n_resets=100):
        for _ in range(n_resets):
            obs, _ = self.reset()
            plt.scatter(obs[0], obs[1])

        plt.grid()
        plt.xlim(self.xlim[0] - 2, self.xlim[1] + 2)
        plt.ylim(self.ylim[0] - 2, self.ylim[1] + 2)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Initial observations")
        plt.show()

    def plot_done_region(self):
        f = plt.figure(figsize=(8, 8))

        x_min, x_max = tuple(np.array(self.xlim))
        y_min, y_max = tuple(np.array(self.ylim))

        img = np.zeros((100, 100))
        for x_idx, x in enumerate(np.linspace(x_max, x_min, 100)):
            for y_idx, y in enumerate(np.linspace(y_max, y_min, 100)):
                obs = np.array([x, y])
                done = self._check_done(obs)

                img[y_idx, x_idx] = 1 if done else 0

        plt.title("Done region")
        plt.imshow(img)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


class PointDynamics(object):
    """
    State: position.
    Action: velocity.
    """

    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        state_next = mu_next + self.sigma * np.random.normal(size=self.s_dim)
        return state_next


class Rectangle(object):
    """
    A helper rectangle class.
    x_start, y_start, refers to the bottom left corner of the rectangle.
    Width is along x direction, which is horizontal.
    Height is along y direction, which is vertical.
    """

    def __init__(self, x_start, y_start, width, height):
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_start + width
        self.y_end = y_start + height
        self.width = width
        self.height = height
        self.x_center = x_start + width * 0.5
        self.y_center = y_start + height * 0.5

    def __str__(self):
        return f"Rectangle(x={self.x_start}, y={self.y_start}, w={self.width}, h={self.height})"

    def calc_distance_to_point(self, px, py):
        """
        Calculates the distance between a point and the rectangle outline
        :param px: x coordinate of point
        :param py: y coordinate of point
        :return: distance
        """
        dx = abs(self.x_center - px) - (self.width * 0.5)
        dy = abs(self.y_center - py) - (self.height * 0.5)
        return np.sqrt((dx * (dx > 0)) ** 2 + (dy * (dy > 0)) ** 2)


if __name__ == "__main__":
    env = PointNavEnv(render_mode="human", task=PointTasks.obstacle)
    seed_everything(1337, env, True)
    obs = env.reset(seed=1337)
    print(obs)
    env.plot_reward()
    env.plot_initial_dist(1000)
    env.plot_done_region()
