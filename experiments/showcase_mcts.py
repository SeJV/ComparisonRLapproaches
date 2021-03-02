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
ROLLOUT_TRAINING = False

training_episodes = 150000
ql_hp = dict()
ql_hp["epsilon"] = 1
ql_hp["epsilon_min"] = .7
ql_hp["epsilon_reduction"] = (ql_hp["epsilon"] - ql_hp["epsilon_min"]) / training_episodes
ql_hp["alpha"] = .01

rollout_policy_agent = QLearningAgent(env, **ql_hp)
if ROLLOUT_TRAINING:
    train_agent(env, rollout_policy_agent, training_episodes=training_episodes)
    rollout_policy_agent.store_models()
else:
    rollout_policy_agent.load_models()

rollout_policy_agent.epsilon = .2
rollout_policy_agent.alpha = .0

mcts_agent = MCTreeSearchAgent(env,
                               playouts_per_action=5,
                               promising_children_playouts=1,
                               rollout_policy_agent=rollout_policy_agent,
                               c=0.1,
                               visualize=True)

state = env.reset()
max_steps = 150
steps = 0
done = False

env.render()
time.sleep(15)

while not done and steps < max_steps:
    action = mcts_agent.act(state)
    state, reward, done, _ = env.step(action)
    mcts_agent.train(state, reward, done)
    steps += 1

env.render()
time.sleep(15)
