from environments import CartPoleEnv
from agent_methods import DeepQNetworkAgent
from train import train_agents
from utils import visualize_training_results_for_agents

"""
In this example comparison three instances of the same kind of agent with different starting alpha values are tested.
Testing contains 5000 episodes of training per agent.  
As testing environment the Cart Pole challenge ist chosen with continuous observation space. 
"""

env = CartPoleEnv()

dqn_agent0003 = DeepQNetworkAgent(env, alpha_start=0.0003, nn_shape=[32, 32], train_size=1024, name='start alpha=0.0003')
dqn_agent001 = DeepQNetworkAgent(env, alpha_start=0.001, nn_shape=[32, 32], train_size=1024, name='start alpha=0.001')
dqn_agent003 = DeepQNetworkAgent(env, alpha_start=0.003, nn_shape=[32, 32], train_size=1024, name='start alpha=0.003')

stats = train_agents(env, [dqn_agent0003, dqn_agent001, dqn_agent003],
                     training_episodes=5000, repetitions=1)
visualize_training_results_for_agents(stats, save_fig='comparison_dqn_cart_pole.png',
                                      train_for='CartPole with Deep Q Network')

