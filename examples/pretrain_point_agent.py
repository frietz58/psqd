import matplotlib
matplotlib.use('Agg')
import argparse

import os
import datetime
from collections import deque, namedtuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algos.psqd import PSQD, PSQDModes

from misc.networks import MLP
from misc.buffer import ReplayBuffer
from misc.utils import seed_everything, PriorityConstraint

from envs.point_nav_env import PointNavEnv, PointTasks

from misc.utils import rollout
from misc.plotting import create_training_plots


if __name__ == "__main__":
    timestamp = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    for iteration in range(1):  # just to repeat experiments n times...

        parser = argparse.ArgumentParser(description='Script for pretraining agents on the 2D point navigation environment.')
        parser.add_argument('--task', help='Which task to pretrain. Either "obstacle", "top_reach" or "side_reach".', default="side_reach")
        args = parser.parse_args()

        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

        # hyperparameters
        SEED = 1 + iteration
        TRAIN_EPS = 2010  # train this many episodes
        GAMMA = 0.99  # discounting
        TAU = 0  # soft polyak target update
        HARD_FREQ = 100
        BATCHSIZE = 256
        AGENT_REWARD_SCALE = 1
        LOSS_FN_STR = "MSELoss"
        MODE = PSQDModes.IS  # importance sampling
        N_VALUE_PARTICLES = 400  # 400 particles in IS grid
        HIDDEN_LAYERS = [32, 32, 32]
        # MODE = PSQDModes.ASVGD  # learn sampler
        # N_VALUE_PARTICLES = 100
        # HIDDEN_LAYERS = [256, 256]
        ACT_FUN_STR = "tanh"
        Q_LR = 0.001
        ZETA_DIM = -1

        LOAD_PRETRAINED_CP = ""
        LOAD_PRETRAINED_EP = ""
        LOAD_BUFFER = ""
        LOAD_REWARD_INFO = ""

        # point env parameters
        if args.task == "obstacle":
            POINT_TASK = PointTasks.obstacle
        elif args.task == "top_reach":
            POINT_TASK = PointTasks.top_reach
        elif args.task == "side_reach":
            POINT_TASK = PointTasks.side_reach
        elif args.task == "obstacle_top":
            POINT_TASK = PointTasks.obstacle_top
        else:
            raise ValueError("Invalid task specified")

        # no constraints during pre-training
        PRIORITY_CONSTRAINTS = []

        env = PointNavEnv(
            task=POINT_TASK,
            detect_collisions=True if len(PRIORITY_CONSTRAINTS) > 0 else False,
        )

        CP = f"../runs/SQL_{MODE.value}"
        CP += f"_{env.__repr__()}"
        CP += timestamp
        CP += f"_withLoadBuffer" if LOAD_BUFFER != "" else ""
        CP += f"_baselineEnvParams"
        # CP += f"_SQL-Comp"  # pre-training on R2 is also what SQL comp does before zero-shot transfer
        CP += f"_batch{BATCHSIZE}"
        CP += f"/iteration{iteration}"
        seed_everything(SEED, env)

        if not os.path.exists(CP):
            os.makedirs(CP)

        PLOT_DIR = os.path.join(CP, "plots")
        if not os.path.exists(PLOT_DIR):
            os.makedirs(PLOT_DIR)

        with open(os.path.join(CP, "hyperparameters.txt"), "w+") as f:
            line_list = [
                f"SEED = {SEED}",
                f"TRAIN_EPS = {TRAIN_EPS}",
                f"GAMMA = {GAMMA}",
                f"TAU = {TAU}",
                f"HARD_FREQ = {HARD_FREQ}",
                f"BATCHSIZE = {BATCHSIZE}",
                f"AGENT_REWARD_SCALE = {AGENT_REWARD_SCALE}",
                f"LOSS_FN_STR = '{LOSS_FN_STR}'",
                f"N_VALUE_PARTICLES = {N_VALUE_PARTICLES}",
                f"SQL_MODE = {MODE.value}",
                f"HIDDEN_LAYERS = {HIDDEN_LAYERS}",
                f"ACT_FUN_STR = '{ACT_FUN_STR}'",
                f"LOAD_PRETRAINED_CP = '{LOAD_PRETRAINED_CP}'",
                f"LOAD_PRETRAINED_EP = {LOAD_PRETRAINED_EP}",
                f"LOAD_BUFFER = '{LOAD_BUFFER}'",
                f"LOAD_REWARD_INFO = '{LOAD_REWARD_INFO}'",
                f"Q_LR = {Q_LR}",
                f"ZETA_DIM = {ZETA_DIM}",
                f"POINT_TASK = {POINT_TASK}",
            ]

            for constraint in PRIORITY_CONSTRAINTS:
                line_list.append(f"PRIORITY_CONSTRAINT = {constraint}")

            f.writelines(line + '\n' for line in line_list)

        TB_WRITER = SummaryWriter(CP, comment="TensorBoardData")

        # create networks for learning the current, lowest priority task
        if LOAD_PRETRAINED_CP:
            q_net = torch.load(f"{LOAD_PRETRAINED_CP}/{LOAD_PRETRAINED_EP}_q_net.pt").to(DEVICE)
            target_q_net = torch.load(f"{LOAD_PRETRAINED_CP}/{LOAD_PRETRAINED_EP}_q_net.pt").to(DEVICE)

            try:
                pi_net = torch.load(f"{LOAD_PRETRAINED_CP}/{LOAD_PRETRAINED_EP}_pi_net.pt").to(DEVICE)
            except FileNotFoundError:
                # this happens when we try to load IS checkpoint, since those don't have policy nets...
                pi_net = None
                if MODE != PSQDModes.IS:
                    raise ValueError("Loading CP while not in IS mode requires that we load from a non-IS CP, aka one with a policy net.")

        else:
            q_net = MLP(
                input_size=env.observation_space.shape[0] + env.action_space.shape[0],
                output_size=1,
                hidden_layers=HIDDEN_LAYERS,
                act_fun_str=ACT_FUN_STR,
            ).to(DEVICE)

            target_q_net = MLP(
                input_size=env.observation_space.shape[0] + env.action_space.shape[0],
                output_size=1,
                hidden_layers=HIDDEN_LAYERS,
                act_fun_str=ACT_FUN_STR,
            ).to(DEVICE)
            target_q_net.load_state_dict(q_net.state_dict())

            pi_net = MLP(
                input_size=env.observation_space.shape[0] + env.action_space.shape[0],
                output_size=env.action_space.shape[0],
                hidden_layers=HIDDEN_LAYERS,
                squash_output=True,
                act_fun_str=ACT_FUN_STR,
            ).to(DEVICE)

        q_nets = []
        q_net_targets = []
        policies = []
        thresholds = []
        for constraint in PRIORITY_CONSTRAINTS:
            q_nets.append(constraint.q_net)
            q_net_targets.append(constraint.q_net)
            thresholds.append(constraint.threshold)

        q_nets.append(q_net)
        q_net_targets.append(target_q_net)

        if MODE == PSQDModes.ASVGD:
            policies.append(pi_net)

        buffer = ReplayBuffer(
            buffer_size=int(1e6),
            batch_size=BATCHSIZE,
            device=DEVICE,
        )

        if LOAD_BUFFER:
            buffer.load(LOAD_BUFFER, reward_info=LOAD_REWARD_INFO)

            if len(PRIORITY_CONSTRAINTS) > 0:
                # discard transitions that violate constraints...
                clean_memory = deque(maxlen=len(buffer))
                experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "info"])

                for i in range(len(buffer)):
                    print(f"filtering buffer: {i}/{len(buffer)}")
                    state = buffer.memory[i].state
                    action = buffer.memory[i].action

                    discard_i = False
                    for constraint in PRIORITY_CONSTRAINTS:
                        # calculate q value for stored transition under the constraint Q
                        transition_q = constraint.q_net(torch.from_numpy(np.concatenate((state, action))).to(DEVICE).to(torch.float32))

                        # get the optimal Q value for the stored transition and see its in constraint threshold
                        if MODE == PSQDModes.IS:
                            # make grid of actions
                            x_, y_ = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(1, -1, 400))
                            actions = np.stack((x_, y_), axis=-1)
                            actions = torch.from_numpy(actions).to(DEVICE).to(torch.float32)
                            q_inp = torch.cat((torch.from_numpy(state).to(DEVICE), actions), dim=-1)
                        elif MODE == PSQDModes.ASVGD:
                            zeta = torch.normal(mean=0.0, std=1.0, size=(32, env.action_space.shape[0])).to(DEVICE).to(torch.float32)
                            states = torch.from_numpy(state).to(DEVICE).repeat(32, 1).to(torch.float32)
                            actions = constraint.pi_net(torch.cat((states, zeta), dim=-1))
                            q_inp = torch.cat((states, actions), dim=-1)
                        else:
                            raise ValueError(f"Unknown mode: {MODE}")

                        qs = constraint.q_net(q_inp)
                        optimal_q = torch.max(qs)

                        # check transition for constraint violation
                        if transition_q - optimal_q + constraint.threshold < 0:
                            discard_i = True
                            break

                    if not discard_i:
                        clean_memory.append(experience(
                            buffer.memory[i].state,
                            buffer.memory[i].action,
                            buffer.memory[i].reward,
                            buffer.memory[i].next_state,
                            buffer.memory[i].done,
                            buffer.memory[i].info))

                print(f"Discarded {len(buffer.memory) - len(clean_memory)} transitions that violated constraints.")
                buffer.memory = clean_memory

        agent = PSQD(
            q_nets=q_nets,
            q_net_targets=q_net_targets,
            asvgd_nets=policies,
            q_lr=Q_LR,
            priority_thresholds=thresholds,
            replay_buffer=buffer,
            device=DEVICE,
            n_particles=N_VALUE_PARTICLES,
            reward_scale=AGENT_REWARD_SCALE,
            gamma=GAMMA,
            tau=TAU,
            hard_freq=HARD_FREQ,
            loss_fn_str=LOSS_FN_STR,
            tensorboard_writer=TB_WRITER,
            action_size=env.action_space.shape[0],
            mode=MODE,
            zeta_dim=ZETA_DIM,
        )

        episiode_rewards = []
        obst_ret_hist = []
        top_ret_hist = []
        side_ret_hist = []
        best_model_avg_reward = -np.inf
        for episode in range(TRAIN_EPS):
            episode_reward, _ = rollout(
                agent,
                env,
                episode,
                seed=SEED,
                n_random_episodes=100,
                log_traj=True,
                ret_all_rewards=True if "PointNav" in env.__repr__() else False,
                mode="train"
            )
            episiode_rewards.append(episode_reward[0] if type(episode_reward) == tuple else episode_reward)
            obst_ret_hist.append(episode_reward[1] if type(episode_reward) == tuple else None)
            top_ret_hist.append(episode_reward[2] if type(episode_reward) == tuple else None)
            side_ret_hist.append(episode_reward[3] if type(episode_reward) == tuple else None)

            avg_reward = np.mean(episiode_rewards[-100:])
            if avg_reward > best_model_avg_reward:
                best_model_avg_reward = avg_reward

                torch.save(agent.q_nets[-1], f"{CP}/best_q_net.pt")
                if MODE == PSQDModes.ASVGD:
                    torch.save(agent.pi_nets[-1], f"{CP}/best_pi_net.pt")

                np.save(f"{CP}/obst_ret_hist.npy", np.array(obst_ret_hist))
                np.save(f"{CP}/top_ret_hist.npy", np.array(top_ret_hist))
                np.save(f"{CP}/side_ret_hist.npy", np.array(side_ret_hist))

            if episode % 100 == 0:
                create_training_plots(agent, env, DEVICE, episode, PLOT_DIR)

                torch.save(agent.q_nets[-1], f"{CP}/{episode}_q_net.pt")
                if MODE == PSQDModes.ASVGD:
                    torch.save(agent.pi_nets[-1], f"{CP}/{episode}_pi_net.pt")

                np.save(f"{CP}/obst_ret_hist.npy", np.array(obst_ret_hist))
                np.save(f"{CP}/top_ret_hist.npy", np.array(top_ret_hist))
                np.save(f"{CP}/side_ret_hist.npy", np.array(side_ret_hist))

        create_training_plots(agent, env, DEVICE, TRAIN_EPS, PLOT_DIR)

        torch.save(agent.q_nets[-1], f"{CP}/{TRAIN_EPS}_q_net.pt")
        if MODE == PSQDModes.ASVGD:
            torch.save(agent.pi_nets[-1], f"{CP}/{TRAIN_EPS}_pi_net.pt")
        buffer.save(CP)
