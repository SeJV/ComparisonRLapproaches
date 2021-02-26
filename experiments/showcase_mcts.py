import time
from environments import MazeEnv
from agents import MCTreeSearchAgent, QLearningAgent
from train import train_agent

"""
The following is a showcase for the ability of Monte Carlo Tree Search to choose an action
due to a action-state tree build and simulations of those nodes. This works also in stochastic 
state transitions. 

It only needs one "training"-episode, because it doesn't train at all. With enough consideration
it can choose the optimal action for given state. 
"""


env = MazeEnv(size='l')

training_episodes = 10000
dql_hp = dict()
dql_hp["epsilon"] = 1
dql_hp["epsilon_min"] = .5
dql_hp["epsilon_reduction"] = (dql_hp["epsilon"] - dql_hp["epsilon_min"]) / training_episodes
dql_hp["alpha"] = .01

rollout_policy_agent = QLearningAgent(env, **dql_hp)
train_agent(env, rollout_policy_agent, training_episodes=training_episodes)

rollout_policy_agent.epsilon = .0
rollout_policy_agent.alpha = .0

mcts_agent = MCTreeSearchAgent(env,
                               playouts_per_action=5,
                               promising_children_playouts=1,
                               rollout_policy_agent=rollout_policy_agent,
                               c=0.05,
                               visualize=True)

state = env.reset()
max_steps = 100
steps = 0
done = False

env.render()
time.sleep(15)

while not done and steps < max_steps:
    action = mcts_agent.act(state)
    state, reward, done, _ = env.step(action)
    print(action)
    mcts_agent.train(state, reward, done)
    steps += 1

env.render()
time.sleep(15)
