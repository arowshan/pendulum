import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1=500, fc2=400, fc3=300):

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Layer (1): 3 inputs (State), fc1 output neurons -> INPUT Layer
        # Layer (2): actions and fc1 input neurons - fc2 output neurons -> HIDDEN Layer 1
        # Layer (3): fc2 input neurons - fc3 output neurons -> HIDDEN Layer 2
        # Layer (4): fc3 input neurons - 1 output (Value) -> OUTPUT Layer
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1 + action_size, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, 1)
        self.init_weights()

    # Initializing the weights to a small non-zero value
    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.05, 0.05)
        self.fc2.weight.data.uniform_(-0.05, 0.05)
        self.fc3.weight.data.uniform_(-0.05, 0.05)
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    # observing state action critic generates the corresponding Q-value
    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.fc4(x)
        return value

