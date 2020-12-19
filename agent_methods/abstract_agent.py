from gym import Env


class AbstractAgent:
    def __init__(self, env: Env, epsilon_start=1.0, epsilon_min=None, alpha_start=0.01, alpha_min=None, name='Agent'):
        self.env = env
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min if epsilon_min else epsilon_start / 100
        self.alpha_start = alpha_start
        self.alpha = alpha_start
        self.alpha_min = alpha_min if alpha_min else alpha_start / 100
        self.name = name

    def reset(self):
        self.epsilon = self.epsilon_start
        self.alpha = self.alpha_start

    def act(self, observation): ...

    def train(self, s_next, reward, done): ...

    def episode_done(self, epsilon_reduction=0, alpha_reduction=0):
        self.epsilon = max(self.epsilon - epsilon_reduction, self.epsilon_min)
        self.alpha = max(self.alpha - alpha_reduction, self.alpha_min)
