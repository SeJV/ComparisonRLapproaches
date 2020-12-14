import numpy as np
from agent_methods import QLearningAgent


class DoubleQLearningAgent(QLearningAgent):
    """
    Double Q-Learning is an variation of the q-learning agent where two independent q-value tables are used.
    To choose an action, the sum of those is used as the single q-table from the q-learning agent.
    To train the two tables, by 50% chance one of the tables will get updated, let's say the first one.
    For the target action for the next state, the position in the table of the second table is used, but the value
    on this position in the first table: q2[s_next, np.argmax(q1[s_next])]
    instead of: np.max(self.q_table[s_next]) in the q-learning agent
    """
    def __init__(self, env, epsilon_start=1.0, epsilon_min=0.0, gamma=0.99, alpha=0.01, name='DoubleQLearningAgent'):
        super().__init__(env, epsilon_start=epsilon_start, epsilon_min=epsilon_min, gamma=gamma, alpha=alpha,
                         name=name)

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
        # Q1(s,a) ← Q1(s,a) + α(reward + γ Q2(s_next, argmax_a Q1(s_next, a))) − Q1(s,a))

        update_q_table = np.random.randint(0, 2)
        q1 = self.q_tables[update_q_table]
        q2 = self.q_tables[1 - update_q_table]

        q1[self.s, self.a] = q1[self.s, self.a] + self.alpha * (
            reward + q2[s_next, np.argmax(q1[s_next])] - q1[self.s, self.a]
        )

        self.q_tables = np.array([q1, q2])
