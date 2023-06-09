import matplotlib
import matplotlib.pyplot as plt

import os
import sys
import io

import numpy as np
import torch
import datetime

from algos.psqd import PSQD, PSQDModes

from misc.utils import seed_everything, PriorityConstraint
from misc.plotting import plot_2D_SQL_value, plot_asvgd_policy, plot_2D_constraint_from_q, plot_2D_global_indifference_space

from envs.franka_panda_env import MujocoPandaEnv, PandaTasks
from misc.utils import rollout
matplotlib.use('TkAgg')  # set rendering backend because it might be overriden in other files during import...


if __name__ == "__main__":

    panda_task = PandaTasks.reach
    env = MujocoPandaEnv(
        xml_path="../envs/panda_scene.xml",
        task=panda_task,
        render_mode="human",
        random_init_qpos=False,
        obs_noise_ratio=0.0,
        episode_length=400,
        verbose=False
    )

    SEED = 1337
    seed_everything(SEED, env)

    avg_reward_dict = {}
    # for load_ep, exp_name in zip(["0", "best"], ["Zeroshot", "Adapted"]):
    for load_ep, exp_name in zip(["best"], ["Adapted"]):
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

        # hyperparameters (most of these are only relevant for training but must be passed to initialize the agent...)
        MODE = PSQDModes.ASVGD
        N_VALUE_PARTICLES = 100  # make sure this matches the CP we are loading below!
        LOAD_PRETRAINED_CP = "../runs/SQL_AmortizedStein_PandaEnv_Reach_Constraints:[Avoid]_2024-03-05_11:42:12_withLoadBuffer_batch32_reproduceFinal"
        LOAD_PRETRAINED_EP = 1000

        # img_dir = LOAD_PRETRAINED_CP + "/imgs"  # slow, but will save each mujoco render image to disk
        img_dir = ""
        if img_dir != "":
            os.makedirs(img_dir, exist_ok=True)

        # priority constraints, list order determined piority order
        PRIORITY_CONSTRAINTS = [
            PriorityConstraint(
                priority_lvl=0,
                name="Avoid",
                threshold=5,
                cp_dir="../runs/SQL_AmortizedStein_PandaEnv_Avoid_Constraints:[]_2024-03-05_11:05:43_32particles_tanh_reproduceFinal",
                load_ep="1500",
                device=DEVICE
            )
        ]

        # loada networks for lowest prio task
        q_net = torch.load(f"{LOAD_PRETRAINED_CP}/{LOAD_PRETRAINED_EP}_q_net.pt").to(DEVICE)
        target_q_net = torch.load(f"{LOAD_PRETRAINED_CP}/{LOAD_PRETRAINED_EP}_q_net.pt").to(DEVICE)

        try:
            pi_net = torch.load(f"{LOAD_PRETRAINED_CP}/{LOAD_PRETRAINED_EP}_pi_net.pt").to(DEVICE)
        except FileNotFoundError:
            # this happens when we try to load IS checkpoint, since those don't have policy nets...
            pi_net = None
            if MODE != PSQDModes.IS:
                raise ValueError("Loading CP while not in IS mode requires that we load from a non-IS CP, aka one with a policy net.")

        q_nets = []
        q_net_targets = []
        policies = []
        thresholds = []
        for constraint in PRIORITY_CONSTRAINTS:
            q_nets.append(constraint.q_net)
            q_net_targets.append(constraint.q_net)
            policies.append(constraint.pi_net)
            thresholds.append(constraint.threshold)

        q_nets.append(q_net)
        q_net_targets.append(target_q_net)
        policies.append(pi_net)

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

        episode_rewards = []
        trajectories = []
        for episode in range(10):

            if img_dir != "":
                img_ep_dir = os.path.join(img_dir, f"episode_{episode}")
                os.makedirs(img_ep_dir, exist_ok=True)
            else:
                img_ep_dir = ""

            # collect trajectories and save each state image...
            reward, traj = rollout(
                agent,
                env,
                episode,
                seed=SEED,
                mode="test",
                log_traj=True,
                img_save_dir=img_ep_dir,
                verbose=False,
            )
            episode_rewards.append(reward)
            trajectories.append(traj)
            print(f"{exp_name} agent, episode {episode} reward: {reward}")

        avg_reward_dict[exp_name] = np.mean(episode_rewards)

        # env.close()

    print("\n===========================================")
    for agent, avg_reward in avg_reward_dict.items():
        print(f"{agent} agent, mean reward: {avg_reward}")
    print("===========================================\n")

