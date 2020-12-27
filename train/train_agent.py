from typing import Dict, List
import numpy as np
from gym import Env
from rl_methods import AbstractAgent


def train_agent(env: Env, agent: AbstractAgent, training_episodes: int = 1000, max_step_per_episode: int = 1000,
                penalty_for_reaching_max_step: float = 1, verbose: bool = True) -> Dict[str, List]:
    """
    Train a single agent for the amount of training steps in a environment. Each training step is one episode.
    Each episode the environment gets reset. Epsilon and alpha of the agent will get linearly reduced to their min
    values.

    :param env: Environment in the openai gym style
    :param agent: instance of a rl method class
    :param training_episodes: amount of episodes trained
    :param max_step_per_episode: in the case of infinitely long running episodes
    :param penalty_for_reaching_max_step: If agent reaches max step, penalty given, by which reward is reduced
    :param verbose: if true, information about steps per episode end, reward as well as epsilon and alpha are presented
    :return: statistics about the training per episode to analyse and visualize
    """
    stats = {'steps': [], 'future_rewards': [], 'epsilon': [], 'alpha': []}
    episode = 1
    running_reward = 0
    for _ in range(training_episodes):
        steps = 0
        reward_sum = 0
        state = env.reset()
        done = False
        while not done and steps <= max_step_per_episode:
            steps += 1
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
            if steps == max_step_per_episode:
                reward -= penalty_for_reaching_max_step
            agent.train(state, reward, done)

        # epsilon min will get reached at the last 20 percentile of training steps
        epsilon_red = (1 / (training_episodes * 0.8)) * (agent.epsilon_start - agent.epsilon_min)
        # alpha min will get reached at the and approached linear
        alpha_red = (1 / training_episodes) * (agent.alpha_start - agent.alpha_min)
        agent.episode_done(epsilon_reduction=epsilon_red, alpha_reduction=alpha_red)

        stats['steps'].append(steps)
        stats['future_rewards'].append(reward_sum)
        stats['epsilon'].append(agent.epsilon)
        stats['alpha'].append(agent.alpha)

        running_reward = 0.9 * running_reward + 0.1 * reward_sum

        if verbose and episode % (training_episodes / 10) == 0:
            print(f'Steps: {round(steps)}, rounded Reward: {round(running_reward, 3)}, '
                  f'Epsilon: {round(agent.epsilon, 3)}, Alpha: {round(agent.alpha, 5)}, ')
        episode += 1

    return stats
