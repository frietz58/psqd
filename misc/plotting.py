import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import kde
from algos.psqd import PSQD, PSQDModes
from misc.utils import PriorityConstraint


def annotate_axis(
        ax: plt.Axes,
        size: int
):
    ax.xaxis.set_major_formatter(
        lambda tick_val, tick_pos: np.around(np.linspace(-size, size, 101)[int(tick_val)], 2))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(
        lambda tick_val, tick_pos: np.around(np.linspace(size, -size, 101)[int(tick_val)], 2))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_2D_SQL_value(
        q_net: torch.nn.Module,
        device: torch.device,
        size: int = 10,
        resolution: int = 100,
        save_path: str = "",
        mode: str = "iter_state",
        state_xy: tuple = (0, 0),
        action_xy: tuple = (0, 0),
        show: bool = False,
        apply_exp: bool = False,
        return_q: bool = False
):
    # since img origin (0, 0) is at top left, we must iterate from max to min and also use matplotlibs [row, col] coordinate system
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if mode == "iter_action":
        # plot Q values iterating over actions
        x_, y_ = np.meshgrid(np.linspace(-size, size, resolution), np.linspace(size, -size, resolution))
        actions = np.stack((x_, y_), axis=-1)
        actions = actions.reshape(-1, 2)
        actions = torch.from_numpy(actions)
        states = np.repeat(np.array([state_xy]), actions.shape[0], 0)
        states = torch.from_numpy(states)
        q_inp = torch.cat((states, actions), 1).to(torch.float32).to(device)
        plt_title = f"Q-value, state={state_xy}"
        plt_xlabel = "action x"
        plt_ylabel = "action y"
    elif mode == "iter_state":
        # plot Q values for action action_xy iterating over states
        x_, y_ = np.meshgrid(np.linspace(-size, size, resolution), np.linspace(size, -size, resolution))
        states = np.stack((x_, y_), axis=-1)
        states = states.reshape(-1, 2)
        states = torch.from_numpy(states)
        actions = np.repeat(np.array([action_xy]), states.shape[0], 0)
        actions = torch.from_numpy(actions)
        q_inp = torch.cat((states, actions), 1).to(torch.float32).to(device)
        plt_title = f"Q-value, action={state_xy}"
        plt_xlabel = "state x"
        plt_ylabel = "state y"
    else:
        raise ValueError(f"mode '{mode}' not recognized")

    if type(q_net) == list:
        q0 = q_net[0](q_inp)
        q1 = q_net[1](q_inp)
        q = q0 + q1
    else:
        q = q_net(q_inp)

    if apply_exp:
        q = torch.exp(q)

    img = q.reshape(resolution, resolution)
    img = ax.imshow(img.detach().cpu().numpy())

    # make ticks correspond to real axis range rather than image coordinates...
    annotate_axis(ax, size)

    # fig.colorbar()
    fig.colorbar(img, cax=cax, orientation='vertical')

    ax.set_title(plt_title)
    ax.set_xlabel(plt_xlabel)
    ax.set_ylabel(plt_ylabel)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

    if show:
        fig.show()

    plt.close(fig)

    if return_q:
        return q


def plot_2D_constraint_from_q(
        q: torch.Tensor,
        constraint: PriorityConstraint,
        resolution: int = 100,
        size: int = 1,
        state_xy: tuple = (0, 0),
        save_path: str = "",
):
    best_q = q.max()
    divergence = q - best_q.repeat(1, q.shape[1])
    c = torch.heaviside(divergence + constraint.threshold, torch.tensor(1, dtype=torch.float32))
    c_img = c.reshape(resolution, resolution)

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    img = ax.imshow(c_img.detach().cpu().numpy(), vmin=0, vmax=1)
    fig.colorbar(img, cax=cax, orientation='vertical')

    annotate_axis(ax, size)
    ax.set_title(f"Constraint {constraint.name}, prio {constraint.priority_lvl}, state={state_xy}")
    ax.set_xlabel("action x")
    ax.set_ylabel("action y")

    if save_path:
        fig.savefig(save_path)

    plt.close(fig)

    return c_img


def plot_2D_global_indifference_space(
        c: list[torch.Tensor],
        q: torch.Tensor,
        save_path: str = "",
):
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    c_prod = np.ones_like(c[0].detach().cpu().numpy())
    for c_img in c:
        constraint_img = np.invert(c_img.detach().cpu().numpy().astype(np.int64)) + 2
        c_prod *= c_img.detach().cpu().numpy().astype(np.int64)
        ax.imshow(constraint_img, cmap="Greys", alpha=constraint_img.astype(float) * 0.75, zorder=1)

    mask = c_prod.astype(bool)
    img = ax.imshow(q.detach().cpu().numpy(), zorder=2, alpha=mask.astype(float))
    fig.colorbar(img, cax=cax, orientation='vertical')

    annotate_axis(ax, 1)
    ax.set_title(f"Global indifference space and pref. Q-values")
    ax.set_xlabel("action x")
    ax.set_ylabel("action y")

    fig.savefig(save_path)
    plt.close(fig)


def plot_env_state(
        env,
        state: tuple,
        save_path: str
):
    env.render_fig, env.render_ax = plt.subplots()  # patch for segmentation fault with closed env render figure...
    obs, info = env.reset(init_state=state)
    env.render(save_path=save_path, ignore_render_mode=True)
    plt.savefig(save_path)
    plt.close()


def plot_point_trajectories_simple(
        agent,
        env,
        grid_res: int = 3,
        save_path: str = "",
):

    env.render_fig, env.render_ax = plt.subplots()
    trajectory_fig = plt.figure()
    trajectory_fig.gca().set_xlim(-10, 10)
    success_counter = 0
    collision_counter = 0

    if env.detect_collisions:
        env.plot_obstacles(trajectory_fig)

    x_, y_ = np.meshgrid(np.linspace(-10, 10, grid_res), np.linspace(-10, 10, grid_res))
    stack_ = np.stack((x_, y_), axis=-1)
    init_poses = stack_.reshape(-1, 2)
    for e in range(init_poses.shape[0]):
        episode_trajectory = []

        try:
            obs, info = env.reset(init_state=init_poses[e])
        except AssertionError:
            # When init_poses[e] is not a valid pos, e.g. inside of an obsacle
            continue

        episode_trajectory.append(obs)
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            action = agent.act(state_tensor=torch.from_numpy(obs).unsqueeze(0).to(agent.device))
            action = action.detach().cpu().numpy().squeeze()
            new_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if done or truncated:
                print("plt", e, np.around(total_reward, 1), info["done_reason"])

            obs = new_obs.squeeze()
            episode_trajectory.append(obs)

        obs_hist = np.array(episode_trajectory)
        if info["done_reason"] == "obstacle collision":
            color = "tab:red"
            collision_counter += 1
        elif info["done_reason"] == "reached goal":
            color = "tab:green"
            success_counter += 1
        else:
            color = "tab:orange"
        trajectory_fig.gca().plot(obs_hist[:, 0], obs_hist[:, 1], marker="o", markersize=4, zorder=1, c=color,
                                  alpha=0.5)

    # plot obstacle rectangles
    if env.task.value == "Obstacle":
        for rect in env.obstacle_patches:
            rectangle = plt.Rectangle((rect.x_start, rect.y_start), rect.width, rect.height, fc='black', ec="black")
            trajectory_fig.gca().add_patch(rectangle)

    trajectory_fig.gca().set_xlim(env.xlim[0], env.xlim[1])
    trajectory_fig.gca().set_ylim(env.ylim[0], env.ylim[1])
    trajectory_fig.suptitle(f"Successes: {success_counter}, colisions: {collision_counter}")
    if save_path:
        plt.savefig(save_path)

    plt.close(trajectory_fig)
    print()


def plot_asvgd_policy(
        agent: PSQD,
        state: np.array,
        n_particles: int,
        save_path: str = "",
        show: bool = False
):
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(torch.float32).to(agent.device)
    state_tensor = state_tensor.unsqueeze(1).repeat(1, n_particles, 1)
    a = agent.asvgd_act(state_tensor, actions_per_state=1)
    a = a.squeeze()
    a_np = a.detach().cpu().numpy()

    x = a_np[:, 0]
    y = a_np[:, 1]

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = 300
    try:
        k = kde.gaussian_kde([x, y])
        # xi, yi = np.mgrid[x.min() - 0.01:x.max() + 0.01:nbins * 1j, y.min() - 0.01:y.max() + 0.01:nbins * 1j]
        xi, yi = np.mgrid[-1:1:nbins * 1j, -1:1:nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # got color from colormesh
        color_value = zi.min()
        cmap = matplotlib.cm.get_cmap('viridis')
        rgba = cmap(color_value)
        plt.gca().set_facecolor(rgba)

        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=cmap,)

    except np.linalg.LinAlgError:
        # when KDR gives singular matrix we just dont plot the KDE
        pass

    plt.scatter(x, y, c="r", s=10)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("action x")
    plt.ylabel("action y")
    plt.title(f"ASVGD policy, state={state}")
    plt.gca().set_aspect('equal')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close()


def create_training_plots(agent, env, device, episode, plot_dir, plot_all_q_nets=False, include_q_sum=False):
    """
    Creates some images that are insightful for the training progress (only for 2D envs due to plotting limitations...)
    """
    if agent.action_size != 2:
        return

    # for state in [np.array([-2, 2]), np.array([6, -2]), np.array([3, -6])]:
    for state in [np.array([0, 0]), np.array([6, 0]), np.array([3, -6]), np.array([0, 2.5])]:
        plot_env_state(env, state, save_path=f"{plot_dir}/env_{state}.png")
        if plot_all_q_nets:
            for i in range(len(agent.q_nets)):
                plot_2D_SQL_value(
                    agent.q_nets[i],
                    device,
                    size=1,
                    mode="iter_action",
                    state_xy=state,
                    save_path=f"{plot_dir}/{episode}_{state}_q{i}.png"
                )
                plot_2D_SQL_value(agent.q_nets[i], device, save_path=f"{plot_dir}/{episode}_q{i}.png")

                if include_q_sum:
                    plot_2D_SQL_value(
                        agent.q_nets,
                        device,
                        size=1,
                        mode="iter_action",
                        state_xy=state,
                        save_path=f"{plot_dir}/{episode}_{state}_qSum.png"
                    )
                    plot_2D_SQL_value(agent.q_nets, device, save_path=f"{plot_dir}/{episode}_qSum.png")

        else:
            plot_2D_SQL_value(
                agent.q_nets[-1],
                device,
                size=1,
                mode="iter_action",
                state_xy=state,
                save_path=f"{plot_dir}/{episode}_{state}_q.png"
            )
            plot_2D_SQL_value(agent.q_nets[-1], device, save_path=f"{plot_dir}/{episode}_q.png")
        if agent.mode == PSQDModes.ASVGD:
            if plot_all_q_nets:
                for i in range(len(agent.pi_nets)):
                    plot_asvgd_policy(agent, state=state, n_particles=100, save_path=f"{plot_dir}/{episode}_{state}_policy{i}.png")
            else:
                plot_asvgd_policy(agent, state=state, n_particles=100, save_path=f"{plot_dir}/{episode}_{state}_policy.png")

    plot_point_trajectories_simple(agent, env, save_path=f"{plot_dir}/{episode}_traj.png")


def plot_env_advantage_log_ws(
        q_nets,
        device,
        save_dir,
        advantage_weight_method,
        weight_fun_str,
        weight_offsets,
        episode=0,
        action=np.array([0, 0]),
        size=10,
        img_res=100
):
    x_, y_ = np.meshgrid(np.linspace(-size, size, img_res), np.linspace(size, -size, img_res))
    states = np.stack((x_, y_), axis=-1)
    states = states.reshape(-1, 2)
    state_tensor = torch.from_numpy(states)  # these are all the states
    actions = np.repeat(np.array([action]), states.shape[0], 0)
    actions = torch.from_numpy(actions)
    q_inp = torch.cat((state_tensor, actions), 1).to(torch.float32).to(device)
    # qs = []
    qs = torch.zeros(q_inp.shape[0], len(q_nets))
    idx = 0
    for q_net in q_nets:
        q = q_net(q_inp)
        q = q.detach()
        qs[:, idx] = q.squeeze()
        idx += 1

    # now for each state we need n actions to approximate the soft value...
    action_range = 1
    grid_res = 20
    action_size = 2
    x_, y_ = np.meshgrid(np.linspace(-action_range, action_range, grid_res),
                         np.linspace(action_range, -action_range, grid_res))
    actions = np.stack((x_, y_), axis=-1)
    actions = actions.reshape(-1, action_size)
    actions = torch.from_numpy(actions).to(device).to(torch.float32)
    states_ = state_tensor.unsqueeze(1).repeat(1, grid_res * grid_res, 1).to(device)
    v_inp = torch.cat((states_, actions.unsqueeze(0).repeat(state_tensor.shape[0], 1, 1)), 2).to(torch.float)

    plt_title = f"log(weight), action={action}"
    plt_xlabel = "state x"
    plt_ylabel = "state y"

    vs = torch.zeros(q_inp.shape[0], len(q_nets))
    idx = 0
    for q_net in q_nets:
        qs_ = q_net(v_inp)

        v = torch.logsumexp(qs_, dim=1)
        v -= torch.log(torch.tensor([grid_res * grid_res])).to(device)  # n = grid_res * grid_res for approximating v
        v = v.detach()
        # v += torch.log(torch.tensor([2]).to(device)) * action_size
        vs[:, idx] = v.squeeze()
        idx += 1

    advantage = qs - vs
    weights = torch.zeros(qs.shape[0], len(q_nets))

    for i in range(len(q_nets)):
        weight = min_log_th(advantage_weight_method(advantage[:, i], level=i, weight_act=weight_fun_str, weight_offsets=weight_offsets))
        weights[:, i] = weight.squeeze()

        img = dcn(weight).reshape(img_res, img_res)
        plt.xlabel(plt_xlabel)
        plt.ylabel(plt_ylabel)
        plt.title(plt_title)
        plt.imshow(img)
        plt.colorbar()
        annotate_axis(plt.gca(), size)
        plt.savefig(f"{save_dir}/{episode}_log(w{i})_action={action}.png")
        plt.close()

        if i > 0:
            weight_sum_img = dcn(weights.sum(1))
            weight_sum_img = weight_sum_img.reshape(img_res, img_res)
            plt.xlabel(plt_xlabel)
            plt.ylabel(plt_ylabel)
            plt.title(plt_title)
            plt.imshow(weight_sum_img)
            plt.colorbar()
            annotate_axis(plt.gca(), size)
            plt.savefig(f"{save_dir}/{episode}_log(w)_sum0:{i}_action={action}.png")
            plt.close()
