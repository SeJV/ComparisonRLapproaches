from gym.envs.toy_text.discrete import DiscreteEnv


class DiscreteAgent:
    def __init__(self, env: DiscreteEnv, epsilon_start=1.0, epsilon_min=0.0, name='Agent'):
        self.action_space = env.action_space.n
        self.state_space = env.observation_space.n
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.name = name

    def reset(self):
        self.epsilon = self.epsilon_start

    def choose_action(self, observation):
        pass

    def train(self, s_next, reward):
        pass

    def episode_done(self, epsilon_reduction=0):
        self.epsilon = max(self.epsilon - epsilon_reduction, self.epsilon_min)
