from gym import wrappers
from environments import PendulumEnv
from agents import DeepDeterministicPolicyGradientAgent
from train import train_agents
from utils import visualize_training_results_for_agents

"""

"""

# Hyperparameters:
training_episodes = 200
reach_min = 1  # 1 means at the end of all training episodes min alpha and epsilon are reached, 0.5 means at 50%

hp = dict()
hp['epsilon'] = 1
hp['epsilon_min'] = 0.01
hp['epsilon_reduction'] = (hp['epsilon'] - hp['epsilon_min']) / (training_episodes * reach_min)
hp['alpha'] = 0.1
hp['alpha_min'] = hp['alpha'] / 10
hp['alpha_reduction'] = (hp['alpha'] - hp['alpha_min']) / (training_episodes * reach_min)
hp['auto_store_models'] = True


env = PendulumEnv()
ddpg_agent = DeepDeterministicPolicyGradientAgent(env, **hp)

stats = train_agents(env, [ddpg_agent], training_episodes=training_episodes, repetitions=1)
visualize_training_results_for_agents(stats, save_fig='ddpg_on_pendulum.png', train_for='Pendulum')

wrapped_env = wrappers.Monitor(env, './monitoring/ddpg', force=True)
state = wrapped_env.reset()
max_steps = 500
steps = 0
done = False

while not done and steps < max_steps:
    action = ddpg_agent.act(state)
    state, _, done, _ = wrapped_env.step(action)
    wrapped_env.render()
    steps += 1

