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
from misc.utils import seed_everything, PriorityConstraint, rollout
from misc.plotting import create_training_plots

from envs.point_nav_env import PointNavEnv, PointTasks
from envs.franka_panda_env import MujocoPandaEnv, PandaTasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for pretraining agents on the mujoco panda environment.')
    parser.add_argument('--task', help='Which task to pretrain. Either "avoid" or "reach".', default="reach")
    args = parser.parse_args()

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    # hyperparameters
    SEED = 2
    TRAIN_EPS = 1500
    GAMMA = 0.99  # discounting
    TAU = 0.005
    HARD_FREQ = 10_000
    BATCHSIZE = 256 if args.task == "avoid" else 32
    AGENT_REWARD_SCALE = 1
    LOSS_FN_STR = "MSELoss"
    MODE = PSQDModes.ASVGD
    N_VALUE_PARTICLES = 32 if args.task == "avoid" else 100
    HIDDEN_LAYERS = [256, 256]
    ACT_FUN_STR = "tanh"
    PI_LR = 0.0001
    Q_LR = 0.001
    ZETA_DIM = -1
    LR_EXPONENTIAL_DECAY = 1  # no LR decay

    LOAD_PRETRAINED_CP = ""
    LOAD_PRETRAINED_EP = "best"
    LOAD_BUFFER = ""
    LOAD_REWARD_INFO = "Avoid_reward"

    N_RANDOM_EPISODES = 0 if LOAD_PRETRAINED_CP else 100

    if args.task == "reach":
        PANDA_TASK = PandaTasks.reach
    elif args.task == "avoid":
        PANDA_TASK = PandaTasks.avoid
    else:
        raise ValueError("Invalid task specified")

    PRIORITY_CONSTRAINTS = [
    ]

    env = MujocoPandaEnv(
        xml_path="../envs/panda_scene.xml",
        task=PANDA_TASK,
        render_mode="",
        episode_length=100 if args.task == "avoid" else 400,
    )

    constraint_names = [constraint.name for constraint in PRIORITY_CONSTRAINTS]
    CP = f"../runs/SQL_{MODE.value}"
    CP += f"_{env.__repr__()}"
    CP += "_Constraints:[" + ">".join(constraint_names) + "]_"
    CP += str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    CP += f"_withLoadBuffer" if LOAD_BUFFER != "" else ""
    CP += f"_{N_VALUE_PARTICLES}particles_{ACT_FUN_STR}"
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
            f"PI_LR = {PI_LR}",
            f"Q_LR = {Q_LR}",
            f"LR_EXPONENTIAL_DECAY = {LR_EXPONENTIAL_DECAY}"
            f"ZETA_DIM = {ZETA_DIM}",
            f"PANDA_TASK = {PANDA_TASK}",
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
        policies.append(constraint.pi_net)
        thresholds.append(constraint.threshold)

    q_nets.append(q_net)
    q_net_targets.append(target_q_net)
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
        pi_lr=PI_LR,
        q_lr=Q_LR,
        lr_exponential_decay=LR_EXPONENTIAL_DECAY,
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
    best_model_avg_reward = -np.inf
    for episode in range(TRAIN_EPS):
        episode_reward, _ = rollout(
            agent,
            env,
            episode,
            seed=SEED,
            n_random_episodes=N_RANDOM_EPISODES,
            log_traj=True,
            mode="train"
        )
        agent.step_lr()
        episiode_rewards.append(episode_reward)

        avg_reward = np.mean(episiode_rewards[-100:])
        if avg_reward > best_model_avg_reward:
            best_model_avg_reward = avg_reward
            print(f"New best model found at episode {episode} with avg reward {avg_reward}")
            torch.save(agent.q_nets[-1], f"{CP}/best_q_net.pt")
            torch.save(agent.pi_nets[-1], f"{CP}/best_pi_net.pt")

        if episode % 100 == 0:
            create_training_plots(agent, env, DEVICE, episode, PLOT_DIR)
            torch.save(agent.q_nets[-1], f"{CP}/{episode}_q_net.pt")
            torch.save(agent.pi_nets[-1], f"{CP}/{episode}_pi_net.pt")

    create_training_plots(agent, env, DEVICE, TRAIN_EPS, PLOT_DIR)
    torch.save(agent.q_nets[-1], f"{CP}/{TRAIN_EPS}_q_net.pt")
    torch.save(agent.pi_nets[-1], f"{CP}/{TRAIN_EPS}_pi_net.pt")
    buffer.save(CP)
