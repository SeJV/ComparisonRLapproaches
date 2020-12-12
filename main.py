import json
import gym
from rl_methods import SarsaAgent, MCControlAgent, DoubleQLearningAgent
from environments import MazeEnv, TAXI, CLIFF_WALKING
from tests import train_agents
from utils import visualize_training_results_for_agents


env = gym.make(TAXI)
mc_control_agent = MCControlAgent(env)
sarsa_agent = SarsaAgent(env, alpha=0.02)
double_q_agent = DoubleQLearningAgent(env, alpha=0.02)

stats = train_agents(env, [sarsa_agent, double_q_agent], training_steps=1000, repetitions=1)

visualize_training_results_for_agents(stats)
