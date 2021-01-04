from environments import FrozenLakeEnv
from rl_methods import MCControlAgent, SarsaAgent, QLearningAgent, DoubleQLearningAgent
from train import train_agents
from utils import visualize_training_results_for_agents

"""
The following is a comparison between four table based discrete RL-Methods. As test environment the 4x4 
frozen lake environment is chosen. Every agent is trained for 10,000 episodes for 3 repeated times. 
The results are visualized in a graph stored in an image. 
"""

env = FrozenLakeEnv(map_name='8x8')

mc_agent = MCControlAgent(env, epsilon_reduction=(1/5000), alpha=0.1, alpha_reduction=(0.1 / 10000))
sarsa_agent = SarsaAgent(env, alpha=0.1)
q_agent = QLearningAgent(env, alpha=0.1)
double_q_agent = DoubleQLearningAgent(env, alpha=0.1)

stats = train_agents(env, [mc_agent], training_episodes=5000, repetitions=1)
visualize_training_results_for_agents(stats, save_fig='table_based_models_frozen_lake.png',
                                      train_for='FrozenLake8x8')

