from environments import FrozenLakeEnv, FROZEN_LAKE_SOLVED_AT
from agents import MCControlAgent, SarsaAgent, QLearningAgent, DoubleQLearningAgent
from train import train_agents
from utils import visualize_training_results_for_agents

"""
The following is a comparison between four table based discrete RL-Methods. As test environment the 4x4 
frozen lake environment is chosen. Every agent is trained for 10,000 episodes for 3 repeated times. 
The results are visualized in a graph stored in an image. 
"""

# Hyperparameters:
training_episodes = 40000

# hp for monte carlo agent(mc)
mc_hp = dict()
mc_hp['epsilon'] = 1.0
mc_hp['epsilon_min'] = 0.01
mc_hp['epsilon_reduction'] = (mc_hp['epsilon'] - mc_hp['epsilon_min']) / (training_episodes * 0.8)
mc_hp['alpha'] = 0.02
mc_hp['alpha_min'] = mc_hp['alpha'] / 10
mc_hp['alpha_reduction'] = (mc_hp['alpha'] - mc_hp['alpha_min']) / (training_episodes * 0.8)

# hp for sarsa agent(s)
s_hp = dict()
s_hp['epsilon'] = 1
s_hp['epsilon_min'] = 0.01
s_hp['epsilon_reduction'] = (s_hp['epsilon'] - s_hp['epsilon_min']) / (training_episodes * 0.7)
s_hp['alpha'] = 0.03
s_hp['alpha_min'] = s_hp['alpha'] / 10
s_hp['alpha_reduction'] = (s_hp['alpha'] - s_hp['alpha_min']) / (training_episodes * 0.7)


# hp for q_agent agent(q)
q_hp = dict()
q_hp['epsilon'] = 1
q_hp['epsilon_min'] = 0.01
q_hp['epsilon_reduction'] = (q_hp['epsilon'] - q_hp['epsilon_min']) / (training_episodes * 0.5)
q_hp['alpha'] = 0.04
q_hp['alpha_min'] = q_hp['alpha'] / 10
q_hp['alpha_reduction'] = (q_hp['alpha'] - q_hp['alpha_min']) / (training_episodes * 0.6)

# hp for double_q_agent agent(dq)
dq_hp = dict()
dq_hp['epsilon'] = 1
dq_hp['epsilon_min'] = 0.01
dq_hp['epsilon_reduction'] = (dq_hp['epsilon'] - dq_hp['epsilon_min']) / (training_episodes * 0.5)
dq_hp['alpha'] = 0.03
dq_hp['alpha_min'] = dq_hp['alpha'] / 5
dq_hp['alpha_reduction'] = (dq_hp['alpha'] - dq_hp['alpha_min']) / (training_episodes * 0.6)


env = FrozenLakeEnv()

mc_agent = MCControlAgent(env, **mc_hp)
sarsa_agent = SarsaAgent(env, **s_hp)
q_agent = QLearningAgent(env, **q_hp)
double_q_agent = DoubleQLearningAgent(env, **dq_hp)

stats = train_agents(env, [mc_agent, sarsa_agent, q_agent, double_q_agent],
                     training_episodes=training_episodes, repetitions=5)
visualize_training_results_for_agents(stats, save_fig='table_based_models_frozen_lake.png',
                                      train_for='FrozenLake', solved_at=FROZEN_LAKE_SOLVED_AT)



