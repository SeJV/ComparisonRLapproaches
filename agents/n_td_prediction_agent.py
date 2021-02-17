from typing import Optional
from gym import Env
import numpy as np
from agents import AbstractAgent
from collections import deque


class NStepTDPredictionAgent(AbstractAgent):
    def __init__(self, env: Env, epsilon: float = 1.0, epsilon_min: float = 0,
                 epsilon_reduction: float = 0.0, alpha: float = 0.01, alpha_min: float = 0,
                 alpha_reduction: float = 0.0, n: int = 2, gamma: float = 0.99, name: str = 'MCControlAgent'):
        super().__init__(env, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_reduction=epsilon_reduction,
                         alpha=alpha, alpha_min=alpha_min, alpha_reduction=alpha_reduction, name=name)
        self.n = n  # n for n-step
        self.gamma = gamma

        # only discrete environments possible
        self.state_space: int = self.env.observation_space.n
        self.action_space: int = self.env.action_space.n

        self.q_table: np.ndarray = np.random.rand(self.state_space, self.action_space) * 0.01

        self.states: deque = deque(maxlen=self.n)
        self.actions: deque = deque(maxlen=self.n)
        self.rewards: deque = deque(maxlen=self.n)

    def reset(self) -> None:
        super().reset()
        self.q_table = np.random.rand(self.state_space, self.action_space) * 0.01

        self.states = deque(maxlen=self.n)
        self.actions = deque(maxlen=self.n)
        self.rewards = deque(maxlen=self.n)

    def act(self, observation: int) -> int:
        if np.random.random() > self.epsilon:
            a = np.argmax(self.q_table[observation])
        else:
            a = np.random.randint(self.action_space)

        self.states.append(observation)
        self.actions.append(a)
        return a

    def train(self, s_next: int, reward: float, done: bool) -> None:
        self.rewards.append(reward)

        if len(self.states) == self.n:
            self._get_g_and_update_q(s_next, done)

        if done:
            while len(self.states) > 1:
                self.states.popleft()
                self.actions.popleft()
                self.rewards.popleft()

                self._get_g_and_update_q(s_next, done)

    def _get_g_and_update_q(self, s_next: float, done: bool):
        g = 0
        discount = 1
        for r in list(self.rewards):
            g += discount * r
            discount *= self.gamma

        q_next = np.max(self.q_table[s_next]) if not done else 0
        g += discount * q_next

        self.q_table[self.states[0], self.actions[0]] += self.alpha * (
            g - self.q_table[self.states[0], self.actions[0]]
        )

    def episode_done(self) -> None:
        super().episode_done()

        self.states = deque(maxlen=self.n)
        self.actions = deque(maxlen=self.n)
        self.rewards = deque(maxlen=self.n)
