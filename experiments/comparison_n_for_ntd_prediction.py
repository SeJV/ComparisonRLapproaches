from environments import FrozenLakeEnv, FROZEN_LAKE_SOLVED_AT
from agents import NStepTDPredictionAgent
from train import train_agents
from utils import visualize_training_results_for_agents

"""
The following is a comparison between four table based discrete RL-Methods. As test environment the 4x4 
frozen lake environment is chosen. Every agent is trained for 10,000 episodes for 3 repeated times. 
The results are visualized in a graph stored in an image. 
"""


env = FrozenLakeEnv()

# Hyperparameters:
training_episodes = 15000

hp = dict()
hp['epsilon'] = 1
hp['epsilon_min'] = 0.01
hp['epsilon_reduction'] = (hp['epsilon'] - hp['epsilon_min']) / (training_episodes * 0.8)
hp['alpha'] = 0.05
hp['alpha_min'] = hp['alpha'] / 3
hp['alpha_reduction'] = (hp['alpha'] - hp['alpha_min']) / (training_episodes * 0.8)


n1 = NStepTDPredictionAgent(env, n=1, name='n1', **hp)
n2 = NStepTDPredictionAgent(env, n=2, name='n2', **hp)
n4 = NStepTDPredictionAgent(env, n=4, name='n4', **hp)
n8 = NStepTDPredictionAgent(env, n=8, name='n8', **hp)

stats = train_agents(env, [n1, n2, n4, n8], training_episodes=training_episodes, repetitions=3, max_step_per_episode=100)
visualize_training_results_for_agents(stats, save_fig='comparison_n_step_td_prediction.png',
                                      train_for='NTD prediction on FrozenLake', solved_at=FROZEN_LAKE_SOLVED_AT)

