import json
import gym
from rl_methods import SarsaAgent, MCControlAgent, DoubleQLearningAgent, DeepQNetworkAgent
from gym.envs.classic_control.cartpole import CartPoleEnv
from tests import train_agents
from utils import visualize_training_results_for_agents


env = CartPoleEnv()
dqn_agent = DeepQNetworkAgent(env, nn_shape=[64], alpha=0.01)

stats = train_agents(env, [dqn_agent], training_steps=50000, repetitions=1)

visualize_training_results_for_agents(stats)
