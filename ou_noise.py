import numpy as np
import random
import copy


class OUNoise:

    def __init__(self, action_size, seed, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_size = action_size
        random.seed(seed)
        self.reset_state()

    def reset_state(self):
        # action space has 1 member only
        self.state = np.array([self.mu])

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for i in range(len(x))]
        )
        self.state = x + dx
        return self.state
