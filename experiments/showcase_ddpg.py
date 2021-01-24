from environments import PendulumEnv
from rl_methods import DeepDeterministicPolicyGradientAgent
from train import train_agents
from utils import visualize_training_results_for_agents

"""

"""

# Hyperparams:
training_episodes = 1000

epsilon = 1.0
epsilon_min = 0.4
epsilon_reduction = (epsilon - epsilon_min) / training_episodes
alpha = 0.001
alpha_min = alpha / 10
alpha_reduction = (alpha - alpha_min) / training_episodes


env = PendulumEnv()
ddpg_agent = DeepDeterministicPolicyGradientAgent(
    env, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_reduction=epsilon_reduction, alpha=alpha,
    alpha_min=alpha_min, alpha_reduction=alpha_reduction, auto_store_models=True)

stats = train_agents(env, [ddpg_agent], training_episodes=training_episodes, repetitions=1)
visualize_training_results_for_agents(stats, save_fig='ddpg_on_pendulum.png', train_for='Pendulum')

