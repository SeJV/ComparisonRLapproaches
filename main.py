from test_environments import CliffWalkingEnv, MazeEnv
from rl_methods import QLearningAgent

cliffEnv = CliffWalkingEnv()
mazeEnv = MazeEnv('s')

env = mazeEnv

agent = QLearningAgent(env)

state = env.reset()
done = False
while not done:
    action = agent.choose_action(state)
    state, reward, done, _ = env.step(action)
    env.render()
    agent.train(state, reward)


