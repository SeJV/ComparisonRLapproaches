from environments import CliffWalkingEnv, MazeEnv
from rl_methods import QLearningAgent, DPAgent

cliffEnv = CliffWalkingEnv()
mazeEnv = MazeEnv('s')

env = mazeEnv

agent = DPAgent(env)
agent.iterative_policy()

state = env.reset()
done = False
while not done:
    action = agent.choose_action(state)
    state, reward, done, _ = env.step(action)
    env.render()
    agent.train(state, reward)


