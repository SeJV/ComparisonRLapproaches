from environments import PendulumEnv
from rl_methods import DeepDeterministicPolicyGradientAgent
from train import train_agents
from utils import visualize_training_results_for_agents

"""

"""


env = PendulumEnv()
ddpg_agent = DeepDeterministicPolicyGradientAgent(env, epsilon=1.0, epsilon_min=0.4, alpha=0.001, alpha_min=0.0001,
                                                  auto_store_models=True)

stats = train_agents(env, [ddpg_agent], training_episodes=5000, repetitions=1)
visualize_training_results_for_agents(stats, save_fig='ddpg_on_pendulum.png', train_for='Pendulum')

