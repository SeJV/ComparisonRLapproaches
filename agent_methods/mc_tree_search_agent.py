import numpy as np
from math import sqrt, log
from agent_methods import AbstractAgent


class _Node:
    def __init__(self, move=None, parent=None):
        self.move = move
        self.parent = parent
        self.N = 0
        self.Q = 0
        self.children = dict()

    def add_children(self, children):
        for child in children:
            self.children[child.move] = child

    @property
    def value(self, explore=0.5):
        if self.N == 0:
            return 0  # alternatively return inf to explore
        else:
            return self.Q / self.N + explore * sqrt(2 * log(self.parent.N) / self.N)


class MCTreeSearchAgent(AbstractAgent):
    def __init__(self, env, gamma=0.99, name='MCControlAgent'):
        super().__init__(env, name=name)
        self.gamma = gamma

        self.root_state = env.reset()
        self.root = _Node()

    def select_node(self):
        node = self.root
        state = self.env.reset()

        while len(node.children) != 0:
            children = node.children.values()


