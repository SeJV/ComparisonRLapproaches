from environments import CliffWalkingEnv, MazeEnv
from rl_methods import QLearningAgent, DoubleQLearningAgent, SarsaAgent, MCControlAgent
from tests import train_agent


cliffEnv = CliffWalkingEnv()
env = cliffEnv

agent0 = QLearningAgent(env, epsilon_start=0.3, epsilon_min=0.1, alpha=0.1)
agent1 = MCControlAgent(env, epsilon_start=0.3, epsilon_min=0.1)

# train_agent(env, agent0, training_steps=3000)
train_agent(env, agent1, training_steps=10000)


