from agent_methods import DeepQNetworkAgent, DeepQNetworkCuriosityAgent
from environments import CartPoleEnv
from train import train_agents
from utils import visualize_training_results_for_agents

env = CartPoleEnv()

dqn_agent = DeepQNetworkAgent(env, nn_shape=[32, 32], alpha=0.001)
dqn_curios_agent = DeepQNetworkCuriosityAgent(env, nn_shape=[32, 32], alpha=0.001)
stats = train_agents(env, [dqn_agent, dqn_curios_agent], training_steps=10000, repetitions=1)

visualize_training_results_for_agents(stats)

