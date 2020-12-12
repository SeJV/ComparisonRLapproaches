import numpy as np
import gym
from rl_methods import SarsaAgent, MCControlAgent, DoubleQLearningAgent, DeepQNetworkAgent
from gym.envs.classic_control.cartpole import CartPoleEnv
from tests import train_agents
from utils import visualize_training_results_for_agents


env = CartPoleEnv()
dqn_agent = DeepQNetworkAgent(env, nn_shape=[16], alpha=0.01, batch_size=1000, gamma=0.9)

stats = train_agents(env, [dqn_agent], training_steps=5000, repetitions=1)

print(dqn_agent.fast_predict(np.array([5, 5, 5, 5])))

visualize_training_results_for_agents(stats)
