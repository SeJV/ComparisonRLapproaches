import numpy as np
from math import sqrt, log
from agent_methods import AbstractAgent


class _Node:
    def __init__(self, move=None, parent=None):
        NotImplementedError()


class MCTreeSearchAgent(AbstractAgent):
    def __init__(self, env, gamma=0.99, name='MCTreeSearchAgent'):
        super().__init__(env, name=name)
        self.gamma = gamma

        self.root_node = _Node()
        NotImplementedError()



