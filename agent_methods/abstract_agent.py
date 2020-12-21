from typing import Optional, Union
import numpy as np
from gym import Env


class AbstractAgent:
    def __init__(self, env: Env, epsilon_start: float = 1.0, epsilon_min: Optional[float] = None,
                 alpha_start: float = 0.01, alpha_min: Optional[float] = None, name: str = 'Agent'):
        self.env = env
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min if epsilon_min else epsilon_start / 100
        self.alpha_start = alpha_start
        self.alpha = alpha_start
        self.alpha_min = alpha_min if alpha_min else alpha_start / 100
        self.name = name

    def reset(self) -> None:
        self.epsilon = self.epsilon_start
        self.alpha = self.alpha_start

    def act(self, observation: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
        ...

    def train(self, s_next: Union[np.ndarray, float, int], reward: float, done: bool) -> None:
        ...

    def episode_done(self, epsilon_reduction: float = 0, alpha_reduction: float = 0) -> None:
        self.epsilon = max(self.epsilon - epsilon_reduction, self.epsilon_min)
        self.alpha = max(self.alpha - alpha_reduction, self.alpha_min)
