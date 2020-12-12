import json

from rl_methods import SarsaAgent
from environments import MazeEnv
from tests import train_agents
from utils import visualize_training_results_for_agents


env = MazeEnv('m')
sarsa_agent = SarsaAgent(env, alpha=0.01)

stats = train_agents(env, [sarsa_agent], training_steps=5000, repetitions=1)

visualize_training_results_for_agents(stats)
