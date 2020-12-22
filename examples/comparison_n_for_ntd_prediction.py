from environments import MazeEnv
from rl_methods import NStepTDPredictionAgent, SarsaAgent
from train import train_agents
from utils import visualize_training_results_for_agents

"""
The following is a comparison between four table based discrete RL-Methods. As test environment the 4x4 
frozen lake environment is chosen. Every agent is trained for 10,000 episodes for 3 repeated times. 
The results are visualized in a graph stored in an image. 
"""

env = MazeEnv('m')

sarsa = SarsaAgent(env, alpha_start=0.01)
n1 = NStepTDPredictionAgent(env, alpha_start=0.1, n=1, name='n1')
n2 = NStepTDPredictionAgent(env, alpha_start=0.1, n=2, name='n2')
n4 = NStepTDPredictionAgent(env, alpha_start=0.1, n=4, name='n4')
n8 = NStepTDPredictionAgent(env, alpha_start=0.1, n=8, name='n8')

stats = train_agents(env, [sarsa], training_episodes=50000, repetitions=1, max_step_per_episode=500)
visualize_training_results_for_agents(stats, save_fig='comparison_n_step_td_prediction.png',
                                      train_for='FrozenLake')

