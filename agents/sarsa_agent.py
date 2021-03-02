from typing import Optional
from gym import Env
import numpy as np
from agents import AbstractAgent


class SarsaAgent(AbstractAgent):
    def __init__(self, env: Env, epsilon: float = 1.0, epsilon_min: float = 0,
                 epsilon_reduction: float = 0.0, alpha: float = 0.01, alpha_min: float = 0,
                 alpha_reduction: float = 0.0, gamma: float = 0.99, name: str = 'Sarsa Agent'):
        super().__init__(env, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_reduction=epsilon_reduction,
                         alpha=alpha, alpha_min=alpha_min, alpha_reduction=alpha_reduction, name=name)
        self.gamma = gamma

        # only discrete environments possible
        self.state_space: int = self.env.observation_space.n
        self.action_space: int = self.env.action_space.n

        self.q_table: np.ndarray = np.random.rand(self.state_space, self.action_space) * 0.01

        self.s = None
        self.a = None
        self.r = None
        self.s_next = None
        self.a_next = None
        self.r_next = None

    def reset(self) -> None:
        super().reset()
        self.q_table = np.random.rand(self.state_space, self.action_space) * 0.01
        self.s = None
        self.a = None
        self.r = None
        self.s_next = None
        self.a_next = None

    def act(self, observation: int) -> int:
        self.s_next = observation

        if np.random.random() > self.epsilon:
            self.a_next = np.argmax(self.q_table[observation])
        else:
            self.a_next = np.random.randint(self.action_space)
        return self.a_next

    def train(self, s_next: int, reward_next: float, done: bool) -> None:
        # s_next stays unused here, in the next choose_action it will become self.s_next
        if self.s is not None and self.a is not None and self.r is not None:
            q_next = self.q_table[self.s_next, self.a_next] if not done else 0

            # Q(s,a) ← Q(s,a) + α(reward + γ Q(s_next, a_next) − Q(s,a))
            self.q_table[self.s, self.a] = self.q_table[self.s, self.a] + self.alpha * (
                    self.r + self.gamma * q_next - self.q_table[self.s, self.a]
            )

        # shifting s_next, a_next and reward_next to s, a and r
        self.s, self.a, self.r = self.s_next, self.a_next, reward_next

