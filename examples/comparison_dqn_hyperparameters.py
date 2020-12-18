from environments import AcrobotEnv
from agent_methods import DeepQNetworkAgent
from train import train_agents
from utils import visualize_training_results_for_agents

env = AcrobotEnv()

dqn_agent0001 = DeepQNetworkAgent(env, alpha=0.0001, nn_shape=[32], train_size=4096, name='DQN alpha=0.0001')
dqn_agent0005 = DeepQNetworkAgent(env, alpha=0.0005, nn_shape=[32], train_size=4096, name='DQN alpha=0.0005')
dqn_agent001 = DeepQNetworkAgent(env, alpha=0.001, nn_shape=[32], train_size=4096, name='DQN alpha=0.001')
dqn_agent002 = DeepQNetworkAgent(env, alpha=0.002, nn_shape=[32], train_size=4096, name='DQN alpha=0.002')

stats = train_agents(env, [dqn_agent0001, dqn_agent0005, dqn_agent001, dqn_agent002],
                     training_steps=1000, repetitions=1)
visualize_training_results_for_agents(stats, save_fig='comparison_dqn_acrobot.png',
                                      environment_name='Acrobot')

