import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib import pyplot as plt

COLORS = [
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.13, 0.52, 0.13],
    [0.5, 0.5, 0.0],
    [0.0, 0.5, 0.5],
    [1, 0.54, 0.0],
    [0.54, 0.17, 0.88],
    [0.54, 0.0, 0.54]
]


def visualize_training_results_for_agents(stats_multiple_agents: dict, save_fig=None, environment_name='environment'):
    for agent_idx, agent_name in enumerate(stats_multiple_agents.keys()):
        steps_per_repetition = []
        rewards_per_repetition = []
        epsilon_per_repetition = []

        for repetition in stats_multiple_agents[agent_name]:
            steps_per_repetition.append(repetition['steps'])
            rewards_per_repetition.append(repetition['rewards'])
            epsilon_per_repetition.append(repetition['epsilon'])

        rewards_per_repetition = np.array(rewards_per_repetition)
        rewards_per_episode = np.swapaxes(rewards_per_repetition, 0, 1)

        length_episode = len(rewards_per_episode)

        smooth_filter = length_episode / 200
        min_rewards_per_episode = gaussian_filter1d(np.min(rewards_per_episode, axis=-1), sigma=smooth_filter)
        mean_rewards_per_episode = gaussian_filter1d(np.mean(rewards_per_episode, axis=-1), sigma=smooth_filter)
        max_rewards_per_episode = gaussian_filter1d(np.max(rewards_per_episode, axis=-1), sigma=smooth_filter)

        color = COLORS[agent_idx]
        color_light = color.copy()
        color_light.append(0.2)
        plt.fill_between(range(length_episode), min_rewards_per_episode, max_rewards_per_episode, color=color_light)
        plt.plot(mean_rewards_per_episode, label=agent_name, color=color, linewidth=3)

    plt.grid()
    plt.ylabel('sum of rewards')
    plt.xlabel('episodes')
    plt.legend(loc='upper left')
    plt.title(f'(rounded) training results for {environment_name}')
    if save_fig:
        plt.savefig(save_fig)
    else:
        plt.show()
