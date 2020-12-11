import numpy as np
from rl_methods import DiscreteAgent


# Off-Policy Monte Carlo-Control
class MCControlAgent(DiscreteAgent):
    def __init__(self, env, epsilon_start=1.0, epsilon_min=0.0, gamma=0.99):
        super().__init__(env, epsilon_start=epsilon_start, epsilon_min=epsilon_min)
        self.gamma = gamma

        self.q_table = np.random.rand(self.state_space, self.action_space) * 0.1
        self.count_table = np.zeros((self.state_space, self.action_space))

        self.states = []
        self.actions = []
        self.rewards = []

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            a = np.argmax(self.q_table[observation])
        else:
            a = np.random.randint(self.action_space)

        self.states.append(observation)
        self.actions.append(a)
        return a

    def train(self, s_next, reward):
        self.rewards.append(reward)

    def episode_done(self, epsilon_reduction=0):
        super().episode_done(epsilon_reduction)

        g = 0
        w = 1
        for t in range(len(self.actions)):
            s_t, a_t, r_t = self.states[t], self.actions[t], self.rewards[t]

            g = self.gamma * g + r_t
            self.count_table[s_t, a_t] += w
            self.q_table[s_t, a_t] += (w / self.count_table[s_t, a_t]) * (g - self.q_table[s_t, a_t])
            if a_t != np.argmax(self.q_table[s_t]):
                break
            w *= 1 / (1 - self.epsilon)  # 1-self.epsilon is not exactly the probability of policy choosing a_t

        self.states = []
        self.actions = []
        self.rewards = []
