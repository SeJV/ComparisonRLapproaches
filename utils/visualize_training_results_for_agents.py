import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib import pyplot as plt

COLORS = [
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.13, 0.52, 0.13],
    [0.54, 0.17, 0.88],
    [1, 0.54, 0.0],
    [0.54, 0.0, 0.54],
    [0.0, 0.5, 0.5],
    [0.5, 0.5, 0.0],
]


def visualize_training_results_for_agents(stats_multiple_agents: dict, save_fig=None, train_for='environment',
                                          zero_zero_start: bool = False) -> None:
    """
    Giving stats resulted from train_agents function, this function creates an plot to showcase those stats.
    The average of reward per episodes is a thick line, the area between minimum and maximum is shown as light and
    transparent color.

    :param stats_multiple_agents: return of train_agents function
    :param save_fig: if None, plot will get shown, else plot is stored as image with <save_fig> as filename
    :param train_for: The title of the plot will say '(rounded) training results for <train_for>'
    :param zero_zero_start: starting the graphs of agents by 0 episodes and 0 reward
    """
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

        lower_quantile_rewards_per_episode = np.quantile(rewards_per_episode, 0.25, axis=-1)
        median_rewards_per_episode = np.median(rewards_per_episode, axis=-1)
        upper_quantile_per_episode = np.quantile(rewards_per_episode, 0.75, axis=-1)

        lower_quantile_smoothed = list(gaussian_filter1d(lower_quantile_rewards_per_episode, sigma=smooth_filter))
        median_rewards_smoothed = list(gaussian_filter1d(median_rewards_per_episode, sigma=smooth_filter))
        upper_quantile_smoothed = list(gaussian_filter1d(upper_quantile_per_episode, sigma=smooth_filter))

        if zero_zero_start:
            lower_quantile_smoothed.insert(0, 0.0)
            median_rewards_smoothed.insert(0, 0.0)
            upper_quantile_smoothed.insert(0, 0.0)

        color = COLORS[agent_idx]
        color_light = color.copy()
        color_light.append(0.2)
        plt.fill_between(range(length_episode), lower_quantile_smoothed, upper_quantile_smoothed, color=color_light)
        plt.plot(median_rewards_smoothed, label=agent_name, color=color, linewidth=3)

    plt.grid()
    plt.ylabel('sum of rewards')
    plt.xlabel('episodes')
    plt.legend(loc='upper left')
    plt.title(f'(rounded) training results for {train_for}')
    if save_fig:
        plt.savefig('plots/' + save_fig)
    else:
        plt.show()
