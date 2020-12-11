import numpy as np
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


def visualize_training_results_for_agents(stats_multiple_agents: dict):
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

        min_rewards_per_episode = np.min(rewards_per_episode, axis=-1)
        mean_rewards_per_episode = np.mean(rewards_per_episode, axis=-1)
        max_rewards_per_episode = np.max(rewards_per_episode, axis=-1)

        length_episode = len(mean_rewards_per_episode)

        color = COLORS[agent_idx]
        color_light = color.append(0.1)
        plt.fill_between(range(length_episode), min_rewards_per_episode, max_rewards_per_episode, color=color_light)
        plt.plot(mean_rewards_per_episode, label=agent_name, color=color, linewidth=3)

    plt.grid()
    plt.legend(loc='lower right')
    plt.title('Trainingsergebnisse')
    plt.show()
