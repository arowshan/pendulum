import random
from collections import deque
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = [state, action, reward, next_state, done]
        self.memory.append(e)

    def sample(self):
        sampling_weights = np.arange(
            1, len(self.memory)+1, 1) / len(self.memory)
        observations = random.choices(
            self.memory, weights=sampling_weights, k=self.batch_size)

        states = (torch.from_numpy(np.vstack(
            np.array(observations)[:, 0])
        ).to(dtype=torch.float32, device=device))

        actions = (torch.from_numpy(np.vstack(
            np.array(observations)[:, 1])).to(dtype=torch.float32, device=device))
        rewards = (torch.from_numpy(np.vstack(
            np.array(observations)[:, 2])).to(dtype=torch.float32, device=device))
        next_states = (torch.from_numpy(np.vstack(
            np.array(observations)[:, 3])).to(dtype=torch.float32, device=device))
        dones = (torch.from_numpy(np.vstack(
            np.array(observations)[:, 4]).astype(np.uint8)).to(dtype=torch.float32, device=device))

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
