import numpy as np


class QLearningAgent:
    def __init__(self, env, gamma=0.99, start_epsilon=1.0, epsilon_min=0.0, alpha=0.01):
        self.action_space = env.action_space.n
        self.state_space = env.observation_space.n
        self.gamma = gamma
        self.start_epsilon = start_epsilon
        self.epsilon = self.start_epsilon
        self.epsilon_min = epsilon_min
        self.alpha = alpha

        self.q_table = np.random.rand(self.state_space, self.action_space) * 0.1

        self.s = None
        self.a = None

    def reset(self):
        self.epsilon = self.start_epsilon
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

    def reduce_epsilon(self, epsilon_reduction=0):
        self.epsilon = max(self.epsilon - epsilon_reduction, 0)
