from environments import MazeEnv
from agents import NStepTDPredictionAgent, SarsaAgent
from train import train_agents
from utils import visualize_training_results_for_agents

"""
The following is a comparison between four table based discrete RL-Methods. As test environment the 4x4 
frozen lake environment is chosen. Every agent is trained for 10,000 episodes for 3 repeated times. 
The results are visualized in a graph stored in an image. 
"""


env = MazeEnv('m')

# Hyperparameters:
training_episodes = 1000

epsilon_start = 1
epsilon_min = 0
epsilon_reduction = (epsilon_start - epsilon_min) / training_episodes

n1 = NStepTDPredictionAgent(env, n=1, alpha=0.1, name='n1', epsilon=epsilon_start, epsilon_min=epsilon_min,
                            epsilon_reduction=epsilon_reduction)
n2 = NStepTDPredictionAgent(env, n=2, alpha=0.1, name='n2', epsilon=epsilon_start, epsilon_min=epsilon_min,
                            epsilon_reduction=epsilon_reduction)
n4 = NStepTDPredictionAgent(env, n=4, alpha=0.1, name='n4', epsilon=epsilon_start, epsilon_min=epsilon_min,
                            epsilon_reduction=epsilon_reduction)
n8 = NStepTDPredictionAgent(env, n=8, alpha=0.1, name='n8', epsilon=epsilon_start, epsilon_min=epsilon_min,
                            epsilon_reduction=epsilon_reduction)

stats = train_agents(env, [n4], training_episodes=training_episodes, repetitions=3, max_step_per_episode=500)
visualize_training_results_for_agents(stats, save_fig='comparison_n_step_td_prediction.png',
                                      train_for='FrozenLake')

