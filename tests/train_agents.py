from typing import List
from gym import Env
from agent_methods import AbstractAgent
from tests import train_agent


def train_agents(env: Env, agents: List[AbstractAgent], training_steps=1000, max_step_per_episode=1000,
                 repetitions=3, verbose=True):
    stats_multiple_agents = dict()
    for agent in agents:
        stats_multiple_agents[agent.name] = []
        for rep in range(repetitions):
            agent.reset()
            stats = train_agent(env, agent, training_steps, max_step_per_episode, verbose)
            stats_multiple_agents[agent.name].append(stats)

    return stats_multiple_agents
