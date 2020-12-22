from typing import List, Dict
from gym import Env
from rl_methods import AbstractAgent
from train import train_agent


def train_agents(env: Env, agents: List[AbstractAgent], training_episodes: int = 1000, max_step_per_episode: int = 1000,
                 repetitions: int = 3, verbose: bool = True) -> Dict[str, List[Dict[str, List]]]:
    """
    Multiple instances of rl methods can be trained here repeatedly. To compare different hyperparameter of the same
    approach or to test the behaviour of different methods. This results in a list of stats from train_agent function
    per agent.

    :param env: Environment to test agents on
    :param agents: list of agents, that each will get trained repeated amounts
    :param training_episodes: training episodes for train_agent function
    :param max_step_per_episode: max steps for train_agent function
    :param repetitions: amount of training repetitions
    :param verbose: verbose information for train_agent function
    :return: list of stats from train_agent function per agent
    """
    stats_multiple_agents = dict()
    for agent in agents:
        stats_multiple_agents[agent.name] = []
        for rep in range(repetitions):
            agent.reset()
            stats = train_agent(env, agent, training_episodes, max_step_per_episode, verbose)
            stats_multiple_agents[agent.name].append(stats)

    return stats_multiple_agents
