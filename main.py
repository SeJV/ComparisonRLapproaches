import numpy as np
import gym
from rl_methods import SarsaAgent, MCControlAgent, DoubleQLearningAgent, DeepQNetworkAgent
from gym.envs.classic_control.mountain_car import MountainCarEnv
from tests import train_agents
from utils import visualize_training_results_for_agents


env = MountainCarEnv()
dqn_agent = DeepQNetworkAgent(env, nn_shape=[64, 64], alpha=0.001, batch_size=2000, gamma=0.9)
# dqn_agent.load()

stats = train_agents(env, [dqn_agent], training_steps=5000, repetitions=1)
visualize_training_results_for_agents(stats)


done = False
state = env.reset()
while not done:
    a = dqn_agent.act(state)
    state, r, done, _ = env.step(a)
    env.render()

