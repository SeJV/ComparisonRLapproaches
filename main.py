from environments import CliffWalkingEnv, MazeEnv
from rl_methods import DoubleQLearningAgent, DoubleQLearningCuriosityAgent, SarsaAgent, MCControlAgent
from tests import train_agent


env = CliffWalkingEnv()

agent = DoubleQLearningCuriosityAgent(env, epsilon_start=0.3, alpha=0.1, icm_scale=50)

train_agent(env, agent, training_steps=3000)


