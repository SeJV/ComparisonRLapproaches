import numpy as np
from rl_methods import QLearningAgent


class DoubleQLearningAgent(QLearningAgent):
    def __init__(self, env, epsilon_start=1.0, epsilon_min=0.0, gamma=0.99, alpha=0.01):
        super().__init__(env, epsilon_start=epsilon_start, epsilon_min=epsilon_min, gamma=gamma, alpha=alpha)

        self.q_tables = np.array([
            np.random.rand(self.state_space, self.action_space) * 0.1,
            np.random.rand(self.state_space, self.action_space) * 0.1
        ])

    def reset(self):
        super().reset()
        self.q_tables = np.array([
            np.random.rand(self.state_space, self.action_space) * 0.1,
            np.random.rand(self.state_space, self.action_space) * 0.1
        ])
        self.s = None
        self.a = None

    def choose_action(self, observation):
        self.s = observation

        if np.random.random() > self.epsilon:
            q_table_sum = np.sum(self.q_tables, axis=0)
            self.a = np.argmax(q_table_sum[observation])
        else:
            self.a = np.random.randint(self.action_space)
        return self.a

    def train(self, s_next, reward):
        # 50% probability the following, otherwise inverse q_tables
        # Q1(s,a) ← Q1(s,a) + α(reward + γ Q2(s_next, argmax_a Q1(s_next, a))) − Q1(s,a))

        update_q_table = np.random.randint(0, 2)
        q1 = self.q_tables[update_q_table]
        q2 = self.q_tables[1 - update_q_table]

        q1[self.s, self.a] = q1[self.s, self.a] + self.alpha * (
            reward + q2[s_next, np.argmax(q1[s_next])] - q1[self.s, self.a]
        )

        self.q_tables = np.array([q1, q2])
