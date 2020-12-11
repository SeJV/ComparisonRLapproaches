from typing import List
from gym.envs.toy_text.discrete import DiscreteEnv
from rl_methods import DiscreteAgent
from tests import train_agent


def train_agents(env: DiscreteEnv, agents: List[DiscreteAgent], training_steps=1000, max_step_per_episode=1000,
                 repetitions=3, verbose=True):
    overall_stats = dict()
    for agent in agents:
        overall_stats[agent.name] = []
        for rep in range(repetitions):
            agent.reset()
            stats = train_agent(env, agent, training_steps, max_step_per_episode, verbose)
            overall_stats[agent].append(stats)

    return overall_stats
