from environments import MazeEnv
from agents import MCTreeSearchAgent, DoubleQLearningAgent
from train import train_agents, train_agent
from utils import visualize_training_results_for_agents

"""
The following is a showcase for the ability of Monte Carlo Tree Search to choose an action
due to a action-state tree build and simulations of those nodes. This works also in stochastic 
state transitions. 

It only needs one "training"-episode, because it doesn't train at all. With enough consideration
it can choose the optimal action for given state. 
"""


env = MazeEnv(size='m')

rollout_policy_agent = DoubleQLearningAgent(env, epsilon=1, epsilon_min=0.6, alpha=0.001)
train_agent(env, rollout_policy_agent, training_episodes=10000)
# this trains actually just to the point, that the agent reaches the goal without finding any treasures

rollout_policy_agent.epsilon = 0.1
rollout_policy_agent.alpha = 0.0001
mcts = MCTreeSearchAgent(env,
                         playouts_per_action=2000,
                         rollout_policy_agent=rollout_policy_agent)
# however, the mcts agent still finds the path to the treasures, because of his action tree

stats = train_agents(env, [mcts], training_episodes=1, repetitions=1, render=True)
visualize_training_results_for_agents(stats, save_fig='mcts_on_medium_maze.png',
                                      train_for='medium Maze', zero_zero_start=True)

