import json
import gym
from rl_methods import SarsaAgent, MCControlAgent, DoubleQLearningAgent
from environments import MazeEnv, CliffWalkingEnv
from tests import train_agents
from utils import visualize_training_results_for_agents

# gyms that can be used with discrete observation space:
# FrozenLake-v0, FrozenLake8x8-v0, CliffWalking-v1?, Taxi-v2

env = CliffWalkingEnv()
mc_control_agent = MCControlAgent(env)
sarsa_agent = SarsaAgent(env, alpha=0.02)
double_q_agent = DoubleQLearningAgent(env, alpha=0.01)

stats = train_agents(env, [sarsa_agent, double_q_agent], training_steps=20000, repetitions=2)

visualize_training_results_for_agents(stats)
