from gym import Env


class AgentType:
    def __init__(self, env: Env, epsilon_start=1.0, epsilon_min=0.0, name='Agent'):
        self.env = env
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.name = name

    def reset(self):
        self.epsilon = self.epsilon_start

    def choose_action(self, observation): ...

    def train(self, s_next, reward): ...

    def episode_done(self, epsilon_reduction=0):
        self.epsilon = max(self.epsilon - epsilon_reduction, self.epsilon_min)
