import numpy as np
from math import sqrt, log
from agent_methods import AbstractAgent


class _Node:
    def __init__(self, move=None, parent=None):
        pass


class MCTreeSearchAgent(AbstractAgent):
    def __init__(self, env, gamma=0.99, name='MCTreeSearchAgent'):
        super().__init__(env, name=name)
        self.gamma = gamma

        self.root_node = _Node()

    def reset(self):
        super().reset()

    def act(self, observation):
        return super().act(observation)

    def train(self, s_next, reward, done):
        super().train(s_next, reward, done)

    def episode_done(self, epsilon_reduction=0, alpha_reduction=0):
        super().episode_done(epsilon_reduction, alpha_reduction)


