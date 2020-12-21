from environments import FrozenLakeEnv
from agent_methods import MCTreeSearchAgent
from train import train_agents
from utils import visualize_training_results_for_agents

"""
The following is a showcase for the ability of Monte Carlo Tree Search to choose an action
due to a action-state tree build and simulations of those nodes. This works also in stochastic 
state transitions. 
"""


env = FrozenLakeEnv()
mcts = MCTreeSearchAgent(env, seconds_per_action=200, amount_simulate_playouts=100)

stats = train_agents(env, [mcts], training_episodes=1, repetitions=1)
visualize_training_results_for_agents(stats, save_fig='mcts_on_small_maze.png',
                                      train_for='small Maze environment')

