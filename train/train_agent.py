from typing import Dict, List
import numpy as np
from gym import Env
from agent_methods import AbstractAgent


def train_agent(env: Env, agent: AbstractAgent, training_steps: int = 1000, max_step_per_episode: int = 1000,
                verbose: bool = True) -> Dict[str, List]:
    stats = {'steps': [], 'rewards': [], 'epsilon': [], 'alpha': []}
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

        # epsilon min will get reached at the last 20 percentile of training steps
        epsilon_red = (1 / (training_steps * 0.8)) * (agent.epsilon_start - agent.epsilon_min)
        # alpha min will get reached at the and approached linear
        alpha_red = (1 / training_steps) * (agent.alpha_start - agent.alpha_min)
        agent.episode_done(epsilon_reduction=epsilon_red, alpha_reduction=alpha_red)

        stats['steps'].append(steps)
        stats['rewards'].append(reward_sum)
        stats['epsilon'].append(agent.epsilon)
        stats['alpha'].append(agent.alpha)

        running_reward = 0.9 * running_reward + 0.1 * reward_sum

        if verbose and episode % (training_steps / 10) == 0:
            print(f'Steps: {round(steps)}, rounded Reward: {round(running_reward, 3)}, '
                  f'Epsilon: {round(agent.epsilon, 3)}, Alpha: {round(agent.alpha, 5)}, ')
        episode += 1

    return stats
