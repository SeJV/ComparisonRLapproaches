import json

from rl_methods import QLearningAgent, DoubleQLearningAgent, DoubleQLearningCuriosityAgent
from environments import MazeEnv
from tests import train_agents
from utils import visualize_training_results_for_agents


env = MazeEnv('l')
q_agent = QLearningAgent(env, alpha=0.1)
double_q_agent = DoubleQLearningAgent(env, alpha=0.1)
double_q_cur_agent = DoubleQLearningCuriosityAgent(env, alpha=0.1, icm_scale=2)

stats = train_agents(env, [q_agent, double_q_agent, double_q_cur_agent], training_steps=5000)

with open('data/multiple_agents.json', 'w') as outfile:
    json.dump(stats, outfile, indent=2)


with open('data/multiple_agents.json') as infile:
    stats = json.load(infile)

visualize_training_results_for_agents(stats)
