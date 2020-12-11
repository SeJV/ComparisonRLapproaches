from environments import CliffWalkingEnv, MazeEnv
from rl_methods import QLearningAgent, DPAgent, QLearningCuriosityAgent, SarsaAgent
from tests import train_agent

cliffEnv = CliffWalkingEnv()
env = cliffEnv

agent = SarsaAgent(env, epsilon_start=0.3, epsilon_min=0.1, alpha=0.1)
agent2 = QLearningAgent(env, epsilon_start=0.3, epsilon_min=0.1, alpha=0.1)

train_agent(env, agent, training_steps=3000)
train_agent(env, agent2, training_steps=3000)


