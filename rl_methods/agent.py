class Agent:
    def __init__(self, env):
        self.action_space = env.action_space.n
        self.state_space = env.observation_space.n

    def reset(self):
        pass

    def choose_action(self, observation):
        pass

    def train(self, s_next, reward):
        pass

    def reduce_epsilon(self, epsilon_reduction=0):
        pass
