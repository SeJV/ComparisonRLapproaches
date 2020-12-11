import numpy as np
from rl_methods import DiscreteAgent


class QLearningAgent(DiscreteAgent):
    def __init__(self, env, epsilon_start=1.0, epsilon_min=0.0, gamma=0.99, alpha=0.01, name='QLearningAgent'):
        super().__init__(env, epsilon_start=epsilon_start, epsilon_min=epsilon_min, name=name)
        self.gamma = gamma
        self.alpha = alpha

        self.q_table = np.random.rand(self.state_space, self.action_space) * 0.1

        self.s = None
        self.a = None

    def reset(self):
        super().reset()
        self.q_table = np.random.rand(self.state_space, self.action_space)
        self.s = None
        self.a = None

    def choose_action(self, observation):
        self.s = observation

        if np.random.random() > self.epsilon:
            self.a = np.argmax(self.q_table[observation])
        else:
            self.a = np.random.randint(self.action_space)
        return self.a

    def train(self, s_next, reward):
        # Q(s,a) ← Q(s,a) + α(reward + γ max(Q(s_next)) − Q(s,a))
        self.q_table[self.s, self.a] = self.q_table[self.s, self.a] + self.alpha * (
            reward + self.gamma * np.max(self.q_table[s_next]) - self.q_table[self.s, self.a]
        )
