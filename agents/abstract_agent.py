from typing import Union
import numpy as np
from gym import Env


class AbstractAgent:
    def __init__(self, env: Env, epsilon: float = 1.0, epsilon_min: float = 0,
                 epsilon_reduction: float = 0, alpha: float = 0.01, alpha_min: float = 0,
                 alpha_reduction: float = 0, name: str = 'Agent'):
        self.env = env
        self.epsilon_start = epsilon
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_reduction = epsilon_reduction
        self.alpha_start = alpha
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_reduction = alpha_reduction
        self.name = name

    def reset(self) -> None:
        self.epsilon = self.epsilon_start
        self.alpha = self.alpha_start

    def act(self, observation: Union[np.ndarray, float, int]) -> Union[np.ndarray, float, int]:
        ...

    def train(self, s_next: Union[np.ndarray, float, int], reward: float, done: bool) -> None:
        ...

    def episode_done(self) -> None:
        self.epsilon = max(self.epsilon - self.epsilon_reduction, self.epsilon_min)
        self.alpha = max(self.alpha - self.alpha_reduction, self.alpha_min)
