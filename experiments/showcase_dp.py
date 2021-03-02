from environments import FrozenLakeEnv
from agents import DPAgent

"""
The following is a showcase for the dynamic programming agent to solve the frozen lake. 
"""

env = FrozenLakeEnv()
dp_agent = DPAgent(env, theta=0.005)
max_steps = 150

reward_sum = 0
iterations = 100
for _ in range(iterations):
    state = env.reset()
    done = False
    while not done:
        action = dp_agent.act(state)
        state, reward, done, _ = env.step(action)
        reward_sum += reward

print(f'Dynamic Programming Agent solved {round(reward_sum / iterations, 2) * 100}% of FrozenLake4x4 iterations')
