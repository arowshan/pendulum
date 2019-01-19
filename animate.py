import gym
import numpy as np
import torch
from ddpg import DDPG
from types import SimpleNamespace
from sklearn.model_selection import ParameterGrid


env = gym.make("Pendulum-v0")
random_seed = 1
env.seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hyperparam_grid = {
    "buffer_size": [int(1e5)],  # replay buffer size
    "batch_size": [1024],       # minibatch size
    "gamma": [0.99],            # discount factor
    "tau": [1e-2],              # tau vs 1- tau weights to update params
    "alpha_actor": [1e-4],      # learning rate of the actor
    "alpha_critic": [1e-3],     # learning rate of the critic
    "weight_decay": [0],        # L2 weight decay
    "mu": [0.0],
    "theta": [0.15],
    "sigma": [0.01],
}
grid = ParameterGrid(hyperparam_grid)
for params in grid:
    params = SimpleNamespace(**params)
    agent = DDPG(
        state_size=3, action_size=1, random_seed=random_seed, hyperparams=params
    )

agent.actor.load_state_dict(torch.load("weights_actor.pth"))
agent.critic.load_state_dict(torch.load("weights_critic.pth"))

state = env.reset()
for i in range(5):
    env.reset()
    for t in range(200):
        action = agent.select_action(state, train=False)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            continue
env.close()
