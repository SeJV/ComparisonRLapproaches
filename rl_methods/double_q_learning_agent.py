from typing import Optional
from gym import Env
import numpy as np
from rl_methods import QLearningAgent


class DoubleQLearningAgent(QLearningAgent):
    def __init__(self, env: Env, epsilon: float = 1.0, epsilon_min: Optional[float] = None,
                 alpha: float = 0.01, alpha_min: Optional[float] = None, gamma: float = 0.99,
                 name: str = 'DoubleQLearningAgent'):
        super().__init__(env, epsilon=epsilon, epsilon_min=epsilon_min, alpha=alpha,
                         alpha_min=alpha_min, gamma=gamma, name=name)

        # only discrete environments possible
        self.state_space = self.env.observation_space.n
        self.action_space = self.env.action_space.n

        self.q_table = np.array([
            np.random.rand(self.state_space, self.action_space) * 0.01,
            np.random.rand(self.state_space, self.action_space) * 0.01
        ])  # two instead of one q-table

    def reset(self) -> None:
        super().reset()
        self.q_table = np.array([
            np.random.rand(self.state_space, self.action_space) * 0.01,
            np.random.rand(self.state_space, self.action_space) * 0.01
        ])

    def act(self, observation: int) -> int:
        self.s = observation

        if np.random.random() > self.epsilon:
            q_table_sum = np.sum(self.q_table, axis=0)
            self.a = np.argmax(q_table_sum[observation])
        else:
            self.a = np.random.randint(self.action_space)
        return self.a

    def train(self, s_next: int, reward: float, done: bool) -> None:
        # 50% probability the following, otherwise switch q_table

        update_q_table = np.random.randint(0, 2)
        q1 = self.q_table[update_q_table]
        q2 = self.q_table[1 - update_q_table]

        q2_next = q2[s_next, np.argmax(q1[s_next])] if not done else 0

        # Q1(s,a) ← Q1(s,a) + α(reward + γ Q2(s_next, argmax_a Q1(s_next, a))) − Q1(s,a))
        q1[self.s, self.a] = q1[self.s, self.a] + self.alpha * (
            reward + q2_next - q1[self.s, self.a]
        )

        self.q_table = np.array([q1, q2])
