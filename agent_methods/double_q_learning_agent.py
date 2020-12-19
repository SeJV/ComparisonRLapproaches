import numpy as np
from agent_methods import QLearningAgent


class DoubleQLearningAgent(QLearningAgent):
    def __init__(self, env, epsilon_start=1.0, epsilon_min=None, alpha_start=0.01, alpha_min=None, gamma=0.99,
                 name='DoubleQLearningAgent'):
        super().__init__(env, epsilon_start=epsilon_start, epsilon_min=epsilon_min, alpha_start=alpha_start,
                         alpha_min=alpha_min, gamma=gamma, name=name)

        # only discrete environments possible
        self.state_space = self.env.observation_space.n
        self.action_space = self.env.action_space.n

        self.q_tables = np.array([
            np.random.rand(self.state_space, self.action_space) * 0.01,
            np.random.rand(self.state_space, self.action_space) * 0.01
        ])

    def reset(self):
        super().reset()
        self.q_tables = np.array([
            np.random.rand(self.state_space, self.action_space) * 0.01,
            np.random.rand(self.state_space, self.action_space) * 0.01
        ])

    def act(self, observation):
        self.s = observation

        if np.random.random() > self.epsilon:
            q_table_sum = np.sum(self.q_tables, axis=0)
            self.a = np.argmax(q_table_sum[observation])
        else:
            self.a = np.random.randint(self.action_space)
        return self.a

    def train(self, s_next, reward, done):
        # 50% probability the following, otherwise switch q_tables

        update_q_table = np.random.randint(0, 2)
        q1 = self.q_tables[update_q_table]
        q2 = self.q_tables[1 - update_q_table]

        q2_next = q2[s_next, np.argmax(q1[s_next])] if not done else 0

        # Q1(s,a) ← Q1(s,a) + α(reward + γ Q2(s_next, argmax_a Q1(s_next, a))) − Q1(s,a))
        q1[self.s, self.a] = q1[self.s, self.a] + self.alpha * (
            reward + q2_next - q1[self.s, self.a]
        )

        self.q_tables = np.array([q1, q2])
