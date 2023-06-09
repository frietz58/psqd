import torch
import numpy as np
import random
from collections import deque, namedtuple
import os


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "info"])

    def add(self, state, action, reward, next_state, done, info={}):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, info)
        self.memory.append(e)

    def sample(self, with_info=False):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)
        infos = [e.info for e in experiences if e is not None]

        if with_info:
            return (states, actions, rewards, next_states, dones, infos)
        else:
            return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def save(self, save_dir):
        print(f"saving buffer to {save_dir}")
        experiences = random.sample(self.memory, k=len(self.memory))  # just sample the entire thing lol
        np.savez(
            os.path.join(save_dir, "buffer.npz"),
            states=np.stack([e.state for e in experiences if e is not None]),
            actions=np.stack([e.action for e in experiences if e is not None]),
            rewards=np.stack([e.reward for e in experiences if e is not None]),
            next_states=np.stack([e.next_state for e in experiences if e is not None]),
            dones=np.stack([e.done for e in experiences if e is not None]),
            infos=np.stack([e.info for e in experiences if e is not None]),
        )

    def load(self, file, reward_info="", relabel_fn=None):
        data = np.load(file, allow_pickle=True)

        state = data["states"]
        action = data["actions"]
        next_state = data["next_states"]
        reward = data["rewards"]
        done = data["dones"]
        info = data["infos"]

        if reward_info:
            # for safety gym, we might want to relabel the transitions with reward form one of the info channels
            for i in range(len(info)):
                reward[i] = info[i][reward_info]

        if relabel_fn is not None:
            for i in range(len(reward)):
                reward[i] = relabel_fn(state[i], action[i])

        for i in range(state.shape[0]):
            print(f"loading buffer: {i}/{state.shape[0]}", end="\r", flush=True)
            self.add(state[i], action[i], reward[i], next_state[i], done[i])

        print()

