from typing import List, Dict
from gym import Env
from agent_methods import AbstractAgent
from train import train_agent


def train_agents(env: Env, agents: List[AbstractAgent], training_steps: int = 1000, max_step_per_episode: int = 1000,
                 repetitions: int = 3, verbose: bool = True) -> Dict[str, List[Dict[str, List]]]:
    stats_multiple_agents = dict()
    for agent in agents:
        stats_multiple_agents[agent.name] = []
        for rep in range(repetitions):
            agent.reset()
            stats = train_agent(env, agent, training_steps, max_step_per_episode, verbose)
            stats_multiple_agents[agent.name].append(stats)

    return stats_multiple_agents
