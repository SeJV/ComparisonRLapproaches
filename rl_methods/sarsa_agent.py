import numpy as np
from rl_methods import Agent


class SarsaAgent(Agent):
    def __init__(self, env, epsilon_start=1.0, epsilon_min=0.0, gamma=0.99, alpha=0.01):
        super().__init__(env=env, epsilon_start=epsilon_start, epsilon_min=epsilon_min)
        self.gamma = gamma
        self.alpha = alpha

        self.q_table = np.random.rand(self.state_space, self.action_space) * 0.1

        self.s = None
        self.a = None
        self.r = None
        self.s_next = None
        self.a_next = None

    def reset(self):
        super().reset()
        self.q_table = np.random.rand(self.state_space, self.action_space)
        self.s = None
        self.a = None
        self.s_next = None
        self.a_next = None

    def choose_action(self, observation):
        self.s_next = observation

        if np.random.random() > self.epsilon:
            self.a_next = np.argmax(self.q_table[observation])
        else:
            self.a_next = np.random.randint(self.action_space)
        return self.a_next

    def train(self, s_next, reward):
        # TODO: something might be wrong, needs deeper analysis
        # s_next stays unused here, in the next choose_action it will become self.s_next
        if self.s and self.a:
            # Q(s,a) ← Q(s,a) + α(reward + γ Q(s_next, a_next) − Q(s,a))
            self.q_table[self.s, self.a] = self.q_table[self.s, self.a] + self.alpha * (
                    reward + self.gamma * self.q_table[self.s_next, self.a_next] - self.q_table[self.s, self.a]
            )

        # shifting s_next and a_next to s and a
        self.s, self.a = self.s_next, self.a_next

