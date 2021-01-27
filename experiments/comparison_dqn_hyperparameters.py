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

# Hyperparameters:
training_episodes = 2000

hp = dict()
hp['epsilon'] = 1
hp['epsilon_min'] = 0.01
hp['epsilon_reduction'] = (hp['epsilon'] - hp['epsilon_min']) / (training_episodes * 0.8)
hp['nn_shape'] = [32, 32]
hp['train_size'] = 1024

hp_alpha_low = hp.copy()
hp_alpha_low['alpha'] = 0.0003
hp_alpha_low['alpha_min'] = hp_alpha_low['alpha'] / 10
hp_alpha_low['alpha_reduction'] = (hp_alpha_low['alpha'] - hp_alpha_low['alpha_min']) / (training_episodes * 0.8)

hp_alpha_mid = hp.copy()
hp_alpha_mid['alpha'] = 0.001
hp_alpha_mid['alpha_min'] = hp_alpha_mid['alpha'] / 10
hp_alpha_mid['alpha_reduction'] = (hp_alpha_mid['alpha'] - hp_alpha_mid['alpha_min']) / (training_episodes * 0.8)

hp_alpha_high = hp.copy()
hp_alpha_high['alpha'] = 0.003
hp_alpha_high['alpha_min'] = hp_alpha_high['alpha'] / 10
hp_alpha_high['alpha_reduction'] = (hp_alpha_high['alpha'] - hp_alpha_high['alpha_min']) / (training_episodes * 0.8)

dqn_agent0003 = DeepQNetworkAgent(env, **hp_alpha_low, name=f'alpha={hp_alpha_low["alpha"]}')
dqn_agent001 = DeepQNetworkAgent(env, **hp_alpha_mid, name=f'alpha={hp_alpha_mid["alpha"]}')
dqn_agent003 = DeepQNetworkAgent(env, **hp_alpha_high, name=f'alpha={hp_alpha_high["alpha"]}')

stats = train_agents(env, [dqn_agent0003, dqn_agent001, dqn_agent003],
                     training_episodes=training_episodes, repetitions=1)
visualize_training_results_for_agents(stats, save_fig='comparison_dqn_cart_pole.png',
                                      train_for='CartPole with Deep Q Network', solved_at=CART_POLE_SOLVED_AT)

