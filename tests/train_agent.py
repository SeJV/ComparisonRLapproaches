import numpy as np
from rl_methods import DiscreteAgent
from gym.envs.toy_text.discrete import DiscreteEnv


def train_agent(env: DiscreteEnv, agent: DiscreteAgent, training_steps=1000, max_step_per_episode=1000,
                verbose=True):
    stats = {'steps': [], 'rewards': [], 'epsilon': []}
    running_reward_sum = 0
    running_steps = 0
    episode = 1
    for _ in range(training_steps):
        steps = 0
        reward_sum = 0
        state = env.reset()
        done = False
        while not done and steps < max_step_per_episode:
            steps += 1
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            reward_sum += reward
            agent.train(state, reward)
        agent.episode_done(epsilon_reduction=(1 / training_steps) * (agent.epsilon_start - agent.epsilon_min))
        running_steps = 0.99 * running_steps + 0.01 * steps
        running_reward_sum = 0.99 * running_reward_sum + 0.01 * reward_sum

        stats['steps'].append(running_steps)
        stats['rewards'].append(running_reward_sum)
        stats['epsilon'].append(agent.epsilon)
        if verbose and episode % (training_steps / 10) == 0:
            print(f'RunningSteps: {round(running_steps)}, RunningReward: {round(running_reward_sum, 3)}, Epsilon: {round(agent.epsilon, 3)}')
        episode += 1

    return stats
