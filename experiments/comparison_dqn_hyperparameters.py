from environments import CartPoleEnv, CART_POLE_SOLVED_AT
from agents import DeepQNetworkAgent
from train import train_agents
from utils import visualize_training_results_for_agents

"""
In this example comparison three instances of the same kind of agent with different starting alpha values are tested.
Testing contains 5000 episodes of training per agent.  
As testing environment the Cart Pole challenge ist chosen with continuous observation space. 
"""

env = CartPoleEnv()

# Hyperparameters (hp):
training_episodes = 3000
reach_min = 1

hp = dict()
hp['epsilon'] = 1
hp['epsilon_min'] = 0.01
hp['epsilon_reduction'] = (hp['epsilon'] - hp['epsilon_min']) / (training_episodes * reach_min)
hp['nn_shape'] = [16]
hp['train_size'] = 256

alpha_reduction_rate = 20

hp_alpha_high = hp.copy()
hp_alpha_high['alpha'] = 0.021
hp_alpha_high['alpha_min'] = hp_alpha_high['alpha'] / alpha_reduction_rate
hp_alpha_high['alpha_reduction'] = (hp_alpha_high['alpha'] - hp_alpha_high['alpha_min']) / (training_episodes * reach_min)

hp_alpha_mid = hp.copy()
hp_alpha_mid['alpha'] = 0.02
hp_alpha_mid['alpha_min'] = hp_alpha_mid['alpha'] / alpha_reduction_rate
hp_alpha_mid['alpha_reduction'] = (hp_alpha_mid['alpha'] - hp_alpha_mid['alpha_min']) / (training_episodes * reach_min)

hp_alpha_low = hp.copy()
hp_alpha_low['alpha'] = 0.019
hp_alpha_low['alpha_min'] = hp_alpha_low['alpha'] / alpha_reduction_rate
hp_alpha_low['alpha_reduction'] = (hp_alpha_low['alpha'] - hp_alpha_low['alpha_min']) / (training_episodes * reach_min)

dqn_agent_high = DeepQNetworkAgent(env, **hp_alpha_high, name=f'alpha={hp_alpha_high["alpha"]}')
dqn_agent_mid = DeepQNetworkAgent(env, **hp_alpha_mid, name=f'alpha={hp_alpha_mid["alpha"]}')
dqn_agent_low = DeepQNetworkAgent(env, **hp_alpha_low, name=f'alpha={hp_alpha_low["alpha"]}')

stats = train_agents(env, [dqn_agent_high, dqn_agent_mid, dqn_agent_low],
                     training_episodes=training_episodes, repetitions=3, max_step_per_episode=550)
visualize_training_results_for_agents(stats, save_fig='comparison_dqn_cart_pole.png',
                                      train_for='CartPole with Deep Q Networks', solved_at=CART_POLE_SOLVED_AT)
