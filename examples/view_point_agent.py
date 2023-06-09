import matplotlib
import matplotlib.pyplot as plt

import os

import numpy as np
import torch
import datetime

from algos.psqd import PSQD, PSQDModes

from misc.utils import seed_everything, PriorityConstraint
from misc.plotting import plot_2D_SQL_value, plot_asvgd_policy, plot_2D_constraint_from_q, plot_2D_global_indifference_space

from envs.point_nav_env import PointNavEnv, PointTasks
from misc.utils import rollout
matplotlib.use('TkAgg')  # set rendering backend because it might be overriden in other files during import...


if __name__ == "__main__":
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    # hyperparameters (most of these are only relevant for training but must be passed to initialize the agent...)
    SEED = 1337
    MODE = PSQDModes.IS
    N_VALUE_PARTICLES = 400  # 400 particles in IS grid
    LOAD_PRETRAINED_CP = "../runs/SQL_ImportanceSampling_PointNavEnv_SideReach_Constraints:[Obstacle>TopReach]_2023-06-26_15:40:47"
    LOAD_PRETRAINED_EP = 1500

    # priority constraints, list order determined piority order
    PRIORITY_CONSTRAINTS = [
        PriorityConstraint(  # highest priority, e.g. obstacle avoidance
            priority_lvl=0,
            name="Obstacle",
            threshold=1,
            cp_dir="../runs/SQL_ImportanceSampling_PointNavEnv_Obstacle_Constraints:[]_2023-06-22_17:13:21",
            load_ep=1500,
            device=DEVICE
        ),

        PriorityConstraint(  # next lowest priority constrained, based on an already priority-adapted q-function/policy
            priority_lvl=1,
            name="TopReach",
            threshold=1,
            cp_dir="../runs/SQL_ImportanceSampling_PointNavEnv_TopReach_Constraints:[Obstacle]_2023-06-26_13:52:42",
            load_ep=1500,
            device=DEVICE
        ),
    ]

    env = PointNavEnv(
        # task=PointTasks.obstacle,
        # task=PointTasks.top_reach,
        task=PointTasks.side_reach,
        detect_collisions=True if len(PRIORITY_CONSTRAINTS) > 0 else False,
        render_mode="human"
    )

    constraint_names = [constraint.name for constraint in PRIORITY_CONSTRAINTS]
    EXP_DIR = f"./SQL_{MODE.value}"
    EXP_DIR += f"_{env.__repr__()}"
    EXP_DIR += "_Constraints:[" + ">".join(constraint_names) + "]_"
    EXP_DIR += str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    seed_everything(SEED, env)

    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)

    with open(os.path.join(EXP_DIR, "hyperparameters.txt"), "w+") as f:
        line_list = [
            f"SEED = {SEED}",
            f"N_VALUE_PARTICLES = {N_VALUE_PARTICLES}",
            f"SQL_MODE = {MODE.value}",
            f"LOAD_PRETRAINED_CP = '{LOAD_PRETRAINED_CP}'",
            f"LOAD_PRETRAINED_EP = {LOAD_PRETRAINED_EP}",
        ]

        for constraint in PRIORITY_CONSTRAINTS:
            line_list.append(f"PRIORITY_CONSTRAINT = {constraint}")

        f.writelines(line + '\n' for line in line_list)

    # loada networks for lowest prio task
    q_net = torch.load(f"{LOAD_PRETRAINED_CP}/{LOAD_PRETRAINED_EP}_q_net.pt").to(DEVICE)
    target_q_net = torch.load(f"{LOAD_PRETRAINED_CP}/{LOAD_PRETRAINED_EP}_q_net.pt").to(DEVICE)

    q_nets = []
    q_net_targets = []
    thresholds = []
    policies = []
    for constraint in PRIORITY_CONSTRAINTS:
        q_nets.append(constraint.q_net)
        q_net_targets.append(constraint.q_net)
        thresholds.append(constraint.threshold)

    q_nets.append(q_net)
    q_net_targets.append(target_q_net)

    agent = PSQD(
        q_nets=q_nets,
        q_net_targets=q_net_targets,
        asvgd_nets=policies,
        priority_thresholds=thresholds,
        replay_buffer=None,
        device=DEVICE,
        n_particles=N_VALUE_PARTICLES,
        action_size=env.action_space.shape[0],
        mode=MODE,
    )

    if env.__class__ == PointNavEnv:
        traj_fig = plt.figure(figsize=(10, 10))
        env.plot_obstacles(traj_fig)

    episode_rewards = []
    trajectories = []
    for episode, init_state in enumerate([
        np.array([-8, -9]), np.array([-4, -9]), np.array([0, -9]), np.array([4, -9]), np.array([8, -9]),
        np.array([-7.5, 0]), np.array([-2, 0]), np.array([0, 0]), np.array([2, 0]), np.array([7.5, 0]),
        # np.array([0, 0])
    ]):
        # collect trajectories and save each state image...
        reward, traj = rollout(
            agent,
            env,
            episode,
            seed=SEED,
            init_state=init_state,
            mode="test",
            log_traj=True,
            img_save_dir=os.path.join(EXP_DIR, str(len(episode_rewards)))
        )
        episode_rewards.append(reward)
        trajectories.append(traj)
        print(f"Episode {episode} reward: {reward}")

        # also plot each Q function for each initial state
        c_imgs = []
        q_imgs = []
        for i in range(len(agent.q_nets)):
            resolution = 100
            q = plot_2D_SQL_value(
                agent.q_nets[i],
                DEVICE,
                size=1,
                mode="iter_action",
                state_xy=init_state,
                save_path=f"{EXP_DIR}/{init_state}_q[{i}].png",
                return_q=True,
                resolution=resolution,
            )
            q_imgs.append(q.reshape(resolution, resolution))
            if i == len(agent.q_nets) - 1:
                break

            c_img = plot_2D_constraint_from_q(
                q,
                PRIORITY_CONSTRAINTS[i],
                save_path=f"{EXP_DIR}/{init_state}_constraint[{i}].png",
            )
            c_imgs.append(c_img)

            # plot global indifference space and q for each initial state
            plot_2D_global_indifference_space(c_imgs, q_imgs[-1], save_path=f"{EXP_DIR}/{init_state}_indifference_space.png")

            # plot initial env states...
            obs, info = env.reset(init_state=init_state)
            env.render(save_path=f"{EXP_DIR}/{init_state}_env.png", ignore_render_mode=True)

    # plot trajectories
    for traj in trajectories:
        for i in range(len(traj) - 1):
            # plot vector from obs to new_obs
            traj_fig.gca().quiver(
                traj[i][0],
                traj[i][1],
                traj[i+1][0] - traj[i][0],
                traj[i+1][1] - traj[i][1],
                angles='xy',
                scale_units='xy',
                scale=1,
                color="k",
                width=0.0035,
                zorder=2
            )
    plt.xlabel("state x")
    plt.ylabel("state y")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.savefig(f"{EXP_DIR}/traj.png")
    plt.close(traj_fig)

    # save rewards
    print(f"\nMean reward: {np.mean(episode_rewards)}")

    env.close()

