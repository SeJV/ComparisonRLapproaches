from environments import CliffWalkingEnv, MazeEnv
from rl_methods import QLearningAgent, DPAgent, QLearningCuriosityAgent

cliffEnv = CliffWalkingEnv()
env = cliffEnv

agent = QLearningCuriosityAgent(env)

state = env.reset()
done = False
while not done:
    action = agent.choose_action(state)
    state, reward, done, _ = env.step(action)
    env.render()
    agent.train(state, reward)


