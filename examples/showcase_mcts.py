from environments import FrozenLakeEnv
from agent_methods import MCTreeSearchAgent, DoubleQLearningAgent
from train import train_agents
from utils import visualize_training_results_for_agents

"""
The following is a showcase for the ability of Monte Carlo Tree Search to choose an action
due to a action-state tree build and simulations of those nodes. This works also in stochastic 
state transitions. 

It only needs one "training"-episode, because it doesn't train at all. With enough consideration
it can choose the optimal action for given state. 
"""


env = FrozenLakeEnv()

rollout_policy_agent = DoubleQLearningAgent(env, epsilon_start=0.3, alpha_start=0.001, gamma=0.99)
mcts = MCTreeSearchAgent(env,
                         amount_test_probability=50,
                         playouts_per_action=20000,
                         promising_children_playouts=100,
                         gamma=0.99,
                         rollout_policy_agent=rollout_policy_agent)

stats = train_agents(env, [mcts], training_episodes=1, repetitions=1, max_step_per_episode=30)
visualize_training_results_for_agents(stats, save_fig='mcts_on_frozen_lake.png',
                                      train_for='Frozen Lake 4x4', zero_zero_start=True)

