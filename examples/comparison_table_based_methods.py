from environments import FrozenLakeEnv
from agent_methods import MCControlAgent, SarsaAgent, QLearningAgent, DoubleQLearningAgent
from train import train_agents
from utils import visualize_training_results_for_agents

env = FrozenLakeEnv()

mc_agent = MCControlAgent(env, alpha_start=0.1)
sarsa_agent = SarsaAgent(env, alpha_start=0.1)
q_agent = QLearningAgent(env, alpha_start=0.1)
double_q_agent = DoubleQLearningAgent(env, alpha_start=0.1)

stats = train_agents(env, [mc_agent, sarsa_agent, q_agent, double_q_agent], training_steps=10000, repetitions=3)
visualize_training_results_for_agents(stats, save_fig='table_based_models_frozen_lake.png',
                                      train_for='FrozenLake')

