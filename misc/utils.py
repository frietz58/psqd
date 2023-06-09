import numpy as np
import torch
import random
import os
from dataclasses import dataclass
import gym

from envs.env_tasks import PointTasks


def rollout(
        agent,
        env: gym.Env,
        episode: int,
        mode: str = "train",
        seed: int = 0,
        init_state: np.array = None,
        log_traj: bool = False,
        ret_all_rewards: bool = False,
        img_save_dir: str = "",
        n_random_episodes: int = 0,
        verbose: bool = True,
        add_prio_reward: bool = False,
        prio_reward_punish: float = 0.0,
        ret_done_reaosn: bool = False,
):
    """
    Runs one episode and, if in training mode, runs a batch update after each step.
    """
    obs, info = env.reset(seed=seed, init_state=init_state)
    done = False
    truncated = False
    total_reward = 0
    trajectory = []
    step_counter = 0
    obst_ret, top_ret, side_ret = 0, 0, 0

    if log_traj:
        trajectory.append(obs)

    while not (done or truncated):
        if episode < n_random_episodes and mode == "train":
            # action = env.action_space.sample()  # this is not deterministic even with seed set...
            action = np.random.uniform(-1, 1, size=agent.action_size)
        else:
            action = agent.act(state_tensor=torch.as_tensor(obs).unsqueeze(0).to(agent.device))
            action = action.detach().cpu().numpy().squeeze()

        new_obs, reward, done, truncated, info = env.step(action)
        if "PointNav" in env.__repr__():
            obst_ret += env.compute_reward(obs, action, task=PointTasks.obstacle)
            top_ret += env.compute_reward(obs, action, task=PointTasks.top_reach)
            side_ret += env.compute_reward(obs, action, task=PointTasks.side_reach)

        if add_prio_reward:
            obs_tensor = torch.as_tensor(obs).float().to(agent.device).unsqueeze(0)
            q_inp, _ = agent._make_q_inp(obs_tensor, batchsize=1)
            states = q_inp[:, :, :2]
            actions = q_inp[:, :, 2:]

            # add selected actions
            states = torch.cat((states, obs_tensor.unsqueeze(0)), dim=1)
            actions = torch.cat((actions, torch.as_tensor(action).unsqueeze(0).unsqueeze(0).float().to(agent.device)), dim=1)

            _, allowed = agent.check_constraints(
                # torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).float().to(agent.device),
                # torch.from_numpy(action).unsqueeze(0).unsqueeze(0).float().to(agent.device)
                states,
                actions
            )
            if not allowed[:, -1].item():
                reward -= prio_reward_punish

        if img_save_dir != "":
            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir)
            img_save_path = os.path.join(img_save_dir, f"{step_counter}.png")
        else:
            img_save_path = ""
        env.render(save_path=img_save_path)

        if "PandaEnv" in env.__repr__() and img_save_path != "":
            # get image from scene
            img = env.mujoco_renderer.render("rgb_array")

            # save image
            from PIL import Image
            img = Image.fromarray(img)
            img.save(img_save_path)

        step_counter += 1
        total_reward += reward

        if log_traj:
            trajectory.append(new_obs)

        # if mode == "train":
        if mode == "train" and episode >= n_random_episodes:
            agent.buffer.add(obs, action, reward, new_obs, done, info)
            agent.update()

        if done or truncated:
            if agent.tb_writer is not None:
                agent.tb_writer.add_scalar("train/reward", total_reward, episode)

            if verbose:
                print(episode, np.around(total_reward, 1), info["done_reason"])

        obs = new_obs

    if ret_all_rewards:
        return (total_reward, obst_ret, top_ret, side_ret), trajectory

    elif log_traj:

        if ret_done_reaosn:
            return total_reward, trajectory, info["done_reason"]
        else:
            return total_reward, trajectory


@dataclass
class PriorityConstraint:
    """
    A task priority constraints.
    It is defined by some pretrained checkpoint and a threshold for the allowed divergence from the optimal action.
    """
    priority_lvl: int  # only here for nice __repr__, priorty order is determined by list order...
    threshold: float
    cp_dir: str
    load_ep: int
    device: torch.device
    name: str

    def __init__(self, priority_lvl, threshold, cp_dir, load_ep, device, name):
        self.priority_lvl = priority_lvl
        self.threshold = threshold
        self.cp_dir = cp_dir
        self.load_ep = load_ep
        self.device = device
        self.name = name

        self.q_net = torch.load(f"{self.cp_dir}/{self.load_ep}_q_net.pt").to(self.device)
        try:
            self.pi_net = torch.load(f"{self.cp_dir}/{self.load_ep}_pi_net.pt").to(self.device)
        except FileNotFoundError:
            self.pi_net = None

    def __repr__(self):
        return f"PriorityConstraint:{self.name}(priority_lvl={self.priority_lvl}, threshold={self.threshold}, cp_dir={self.cp_dir}, load_ep={self.load_ep})"


def se_kernel(x, y=0, sigma=1, lengthscale=1):
    try:
        # if x is a tensor
        x = x.cpu().detach().numpy()
    except AttributeError:
        pass

    return sigma**2 * np.exp(- ((x - y)**2 / 2*lengthscale**2))


def min_log_th(t: torch.Tensor):
    """
    np.log(x) return nan if x < 0
    this function return -3.4028235e+38 when x <= 0
    """
    mask = (t <= 0)
    notmask = ~mask
    out = t.clone()
    out[notmask] = torch.log(out[notmask])
    out[mask] = torch.finfo(torch.float32).min
    return out


def asvgd_rbf_kernel(input_1, input_2):
    k_fix, out_dim1 = input_1.size()[-2:]
    k_upd, out_dim2 = input_2.size()[-2:]
    assert out_dim1 == out_dim2

    # Compute the pairwise distances of left and right particles.
    diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
    dist_sq = diff.pow(2).sum(-1)
    dist_sq = dist_sq.unsqueeze(-1)

    # Get median.
    median_sq = torch.median(dist_sq, dim=1)[0]
    median_sq = median_sq.unsqueeze(1)

    # h = median_sq / np.log(k_fix + 1.) + .001
    h = median_sq / (np.log(k_fix + 1.) + .001)
    # h = torch.autograd.Variable(h, requires_grad=False)

    kappa = torch.exp(-dist_sq / h)

    # Construct the gradient
    kappa_grad = -2. * diff / h * kappa
    return kappa, kappa_grad


def adaptive_isotropic_gaussian_kernel(xs, ys, h_min=1e-3):
    """
    Taken from SQL reference implementation: https://github.com/haarnoja/softqlearning/blob/master/softqlearning/

    Gaussian kernel with dynamic bandwidth.

    The bandwidth is adjusted dynamically to match median_distance / log(Kx).
    See [2] for more information.

    Args:
        xs (`torch.Tensor`): A tensor of shape (N x Kx x D) containing N sets of Kx
            particles of dimension D. This is the first kernel argument.
        ys (`torch.Tensor`): A tensor of shape (N x Ky x D) containing N sets of Kx
            particles of dimension D. This is the second kernel argument.
        h_min (`float`): Minimum bandwidth.

    Returns:
        `dict`: Returned dictionary has two fields:
            'output': A `torch.Tensor` object of shape (N x Kx x Ky) representing
                the kernel matrix for inputs `xs` and `ys`.
            'gradient': A `torch.Tensor` object of shape (N x Kx x Ky x D)
                representing the gradient of the kernel with respect to `xs`.

    Reference:
        [2] Qiang Liu,Dilin Wang, "Stein Variational Gradient Descent: A General
            Purpose Bayesian Inference Algorithm," Neural Information Processing
            Systems (NIPS), 2016.
    """
    Kx, D = xs.size()[-2:]
    Ky, D2 = ys.size()[-2:]
    assert D == D2

    leading_shape = xs.size()[:-2]

    # Compute the pairwise distances of left and right particles.
    diff = xs.unsqueeze(-2) - ys.unsqueeze(-3)
    # ... x Kx x Ky x D

    dist_sq = torch.sum(diff**2, dim=-1)
    # ... x Kx x Ky

    # Get median.
    # input_shape = torch.cat((leading_shape, torch.tensor([Kx * Ky])), dim=0)
    input_shape = leading_shape + (Kx * Ky,)
    values, _ = torch.topk(
        input=dist_sq.view(input_shape),
        k=(Kx * Ky // 2 + 1),  # This is exactly true only if Kx*Ky is odd.
        dim=-1,
        largest=True)  # ... x floor(Ks*Kd/2)

    medians_sq = values[..., -1]  # ... (shape) (last element is the median)

    h = medians_sq / torch.log(torch.tensor(Kx, dtype=medians_sq.dtype))  # ... (shape)
    h = torch.maximum(h, torch.tensor(h_min, dtype=h.dtype))
    h = torch.autograd.Variable(h, requires_grad=False)  # Just in case.
    h_expanded_twice = h.unsqueeze(-1).unsqueeze(-1)
    # ... x 1 x 1

    kappa = torch.exp(-dist_sq / h_expanded_twice)  # ... x Kx x Ky

    # Construct the gradient
    h_expanded_thrice = h_expanded_twice.unsqueeze(-1)
    # ... x 1 x 1 x 1
    kappa_expanded = kappa.unsqueeze(-1)  # ... x Kx x Ky x 1

    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded
    # ... x Kx x Ky x D

    return kappa, kappa_grad


def seed_everything(seed, env, printing=False):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.action_space.np_random.seed(seed)
        env.observation_space.np_random.seed(seed)
        env.seed(seed)
    except AttributeError:
        print("env seeding had error...")
        pass
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if printing:
        print(random.uniform(0, 1))
        print(np.random.uniform(0, 1))
        print(torch.rand(1))
        print(env.action_space.sample())
        print(env.observation_space.sample())
