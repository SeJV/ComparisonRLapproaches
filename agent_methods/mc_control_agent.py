from typing import Optional
from gym import Env
import numpy as np
from agent_methods import AbstractAgent


class MCControlAgent(AbstractAgent):
    def __init__(self, env: Env, epsilon_start: float = 1.0, epsilon_min: Optional[float] = None,
                 alpha_start: float = 0.01, alpha_min: Optional[float] = None, gamma: float = 0.99,
                 name: str = 'MCControlAgent'):
        super().__init__(env, epsilon_start=epsilon_start, epsilon_min=epsilon_min, alpha_start=alpha_start,
                         alpha_min=alpha_min, name=name)
        self.gamma = gamma

        # only discrete environments possible
        self.state_space = self.env.observation_space.n
        self.action_space = self.env.action_space.n

        self.q_table = np.zeros((self.state_space, self.action_space))

        self.states = []
        self.actions = []
        self.rewards = []

    def reset(self) -> None:
        super().reset()
        self.q_table = np.zeros((self.state_space, self.action_space))

        self.states = []
        self.actions = []
        self.rewards = []

    def act(self, observation: int) -> int:
        if np.random.random() > self.epsilon:
            a = np.argmax(self.q_table[observation])
        else:
            a = np.random.randint(self.action_space)

        self.states.append(observation)
        self.actions.append(a)
        return a

    def train(self, s_next: int, reward: float, done: bool) -> None:
        self.rewards.append(reward)

    def episode_done(self, epsilon_reduction: float = 0, alpha_reduction: float = 0) -> None:
        super().episode_done(epsilon_reduction, alpha_reduction)

        g = 0
        for t in reversed(range(len(self.actions))):
            s_t, a_t, r_t = self.states[t], self.actions[t], self.rewards[t]
            g = self.gamma * g + r_t
            self.q_table[s_t, a_t] = self.q_table[s_t, a_t] + self.alpha * (g - self.q_table[s_t, a_t])

        self.states = []
        self.actions = []
        self.rewards = []
