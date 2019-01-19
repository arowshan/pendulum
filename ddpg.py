import numpy as np
import random
import copy
from actor import Actor
from critic import Critic
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG:
    def __init__(self, state_size, action_size, random_seed, hyperparams):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.hyperparams = hyperparams

        self.actor = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_noise = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_optim = optim.Adam(
            self.actor.parameters(), lr=hyperparams.alpha_actor
        )

        self.critic = Critic(
            state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(
            state_size, action_size, random_seed).to(device)
        self.critic_optim = optim.Adam(
            self.critic.parameters(),
            lr=hyperparams.alpha_critic,
            weight_decay=hyperparams.weight_decay,
        )

        self.replay_buffer = ReplayBuffer(
            hyperparams.buffer_size, hyperparams.batch_size, random_seed
        )

        self.noise = OUNoise(
            action_size,
            random_seed,
            self.hyperparams.mu,
            self.hyperparams.theta,
            self.hyperparams.sigma,
        )

    def step(self, state, action, reward, next_state, done):

        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) > self.hyperparams.batch_size:
            observations = self.replay_buffer.sample()
            self.update_params(observations)

    def select_action(self, state, train=True, nn_noise=False):
        state = torch.from_numpy(state).to(dtype=torch.float32, device=device)
        self.actor.eval()
        if nn_noise:
            action = self.actor_noise(state).cpu().data.numpy()
        else:
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if train:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset_state()

    def update_params(self, observations):

        states, actions, rewards, next_states, dones = observations
        next_actions = self.actor_target(next_states)
        next_Q_values = self.critic_target(next_states, next_actions)
        Q_values = rewards + (self.hyperparams.gamma *
                              next_Q_values * (1 - dones))

        expected_Q = self.critic(states, actions)
        Q_values_loss = F.l1_loss(expected_Q, Q_values)
        self.critic_optim.zero_grad()
        Q_values_loss.backward()
        self.critic_optim.step()

        policy_loss = -self.critic(states, self.actor(states))
        policy_loss = policy_loss.mean()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        for qtarget_param, qlocal_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            qtarget_param.data.copy_(
                self.hyperparams.tau * qlocal_param.data + (1.0 - self.hyperparams.tau) * qtarget_param.data)

        for target_param, local_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                self.hyperparams.tau * local_param.data + (1.0 - self.hyperparams.tau) * target_param.data)
