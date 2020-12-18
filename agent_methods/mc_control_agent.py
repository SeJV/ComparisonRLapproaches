import numpy as np
from agent_methods import AbstractAgent


class MCControlAgent(AbstractAgent):
    def __init__(self, env, epsilon_start=1.0, epsilon_min=0.0, gamma=0.99, alpha=0.01, name='MCControlAgent'):
        super().__init__(env, epsilon_start=epsilon_start, epsilon_min=epsilon_min, name=name)
        self.gamma = gamma
        self.alpha = alpha

        # only discrete environments possible
        self.state_space = self.env.observation_space.n
        self.action_space = self.env.action_space.n

        self.q_table = np.zeros((self.state_space, self.action_space))

        self.states = []
        self.actions = []
        self.rewards = []

    def reset(self):
        super().reset()
        self.q_table = np.zeros((self.state_space, self.action_space))

        self.states = []
        self.actions = []
        self.rewards = []

    def act(self, observation):
        if np.random.random() > self.epsilon:
            a = np.argmax(self.q_table[observation])
        else:
            a = np.random.randint(self.action_space)

        self.states.append(observation)
        self.actions.append(a)
        return a

    def train(self, s_next, reward, done):
        self.rewards.append(reward)

    def episode_done(self, epsilon_reduction=0):
        super().episode_done(epsilon_reduction)

        g = 0
        for t in reversed(range(len(self.actions))):
            s_t, a_t, r_t = self.states[t], self.actions[t], self.rewards[t]
            g = self.gamma * g + r_t
            self.q_table[s_t, a_t] = self.q_table[s_t, a_t] + self.alpha * (g - self.q_table[s_t, a_t])

        self.states = []
        self.actions = []
        self.rewards = []
