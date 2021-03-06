from typing import Optional
from gym import Env
import numpy as np
from agents import AbstractAgent


class QLearningAgent(AbstractAgent):
    def __init__(self, env: Env, epsilon: float = 1.0, epsilon_min: float = 0,
                 epsilon_reduction: float = 0.0, alpha: float = 0.01, alpha_min: float = 0,
                 alpha_reduction: float = 0.0, gamma: float = 0.99, name: str = 'Q-Learning Agent'):
        super().__init__(env, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_reduction=epsilon_reduction,
                         alpha=alpha, alpha_min=alpha_min, alpha_reduction=alpha_reduction, name=name)
        self.gamma = gamma

        # only discrete environments possible
        self.state_space: int = self.env.observation_space.n
        self.action_space: int = self.env.action_space.n

        self.q_table: np.ndarray = np.random.rand(self.state_space, self.action_space) * 0.01
        self.q_table_file = f'models/{self.name}/q_table.npy'

        self.s: Optional[int] = None
        self.a: Optional[int] = None

    def reset(self) -> None:
        super().reset()
        self.q_table = np.random.rand(self.state_space, self.action_space) * 0.01
        self.s = None
        self.a = None

    def act(self, observation: int) -> int:
        self.s = observation

        if np.random.random() > self.epsilon:
            self.a = np.argmax(self.q_table[observation])
        else:
            self.a = np.random.randint(self.action_space)
        return self.a

    def train(self, s_next: int, reward: float, done: bool) -> None:
        q_next = np.max(self.q_table[s_next]) if not done else 0

        # Q(s,a) ← Q(s,a) + α(reward + γ max(Q(s_next)) − Q(s,a))
        self.q_table[self.s, self.a] = self.q_table[self.s, self.a] + self.alpha * (
            reward + self.gamma * q_next - self.q_table[self.s, self.a]
        )

    def get_state_value(self, state: int) -> float:
        return np.max(self.q_table[state])

    def store_models(self) -> None:
        np.save(self.q_table_file, self.q_table)

    def load_models(self) -> None:
        self.q_table = np.load(self.q_table_file)

