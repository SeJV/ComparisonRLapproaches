from gym import wrappers
from environments import PendulumEnv
from agents import DeepDeterministicPolicyGradientAgent
from train import train_agent

# Hyperparameters:
training_episodes = 3000
reach_min = 1  # 1 means at the end of all training episodes min alpha and epsilon are reached, 0.5 means at 50%

env = PendulumEnv()
DDPG_TRAINING = True

hp = dict()
hp['epsilon'] = 0.5
hp['epsilon_min'] = 0.01
hp['epsilon_reduction'] = (hp['epsilon'] - hp['epsilon_min']) / (training_episodes * reach_min)
hp['alpha'] = 0.005
hp['alpha_min'] = hp['alpha'] / 5
hp['alpha_reduction'] = (hp['alpha'] - hp['alpha_min']) / (training_episodes * reach_min)
hp['auto_store_models'] = True

ddpg_agent = DeepDeterministicPolicyGradientAgent(env, **hp)
ddpg_agent.load_models()

if DDPG_TRAINING:
    train_agent(env, ddpg_agent, training_episodes=training_episodes)

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

