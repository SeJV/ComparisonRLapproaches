import numpy as np
from gym import Env
from rl_methods import AbstractAgent


def train_agent(env: Env, agent: AbstractAgent, training_steps=1000, max_step_per_episode=1000,
                verbose=True):
    stats = {'steps': [], 'rewards': [], 'epsilon': []}
    episode = 1
    running_reward = 0
    for _ in range(training_steps):
        steps = 0
        reward_sum = 0
        state = env.reset()
        done = False
        while not done and steps < max_step_per_episode:
            steps += 1
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
            agent.train(state, reward, done)

        agent.episode_done(epsilon_reduction=(1 / training_steps) * (agent.epsilon_start - agent.epsilon_min))

        stats['steps'].append(steps)
        stats['rewards'].append(reward_sum)
        stats['epsilon'].append(agent.epsilon)

        running_reward = 0.9 * running_reward + 0.1 * reward_sum

        if verbose and episode % (training_steps / 10) == 0:
            print(f'Steps: {round(steps)}, rounded Reward: {round(running_reward, 3)}, Epsilon: {round(agent.epsilon, 3)}')
        episode += 1

    return stats
