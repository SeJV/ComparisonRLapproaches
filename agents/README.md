# Agents

## Table of Contents
- [Abstract Agent](/agents/abstract_agent.py)
- [Dynamic Programming](/agents/dynamic_programming_agent.py)
- [Monte Carlo Control](/agents/mc_control_agent.py)
- [Sarsa](/agents/sarsa_agent.py)
- [N-Step TD Control](/agents/n_td_prediction_agent.py)
- [Q-Learning](/agents/q_learning_agent.py)
- [Double Q-Learning](/agents/double_q_learning_agent.py)
- [Deep Q-Network](/agents/deep_q_network_agent.py)
- [Deep Q-Network with Curiosity](/agents/deep_q_network_curiosity_agent.py)
- [Monte Carlo Tree Search](/agents/mc_tree_search_agent.py)
- [Deep Deterministic Policy Gradient](/agents/deep_deterministic_policy_gradient_agent.py)

### Abstract Agent
Agent classes are implementations for some methods found in literature for approaches to solve
reinforcement learning problems. Methods without further sources follow closely [Reinforcement Learning - An
Introduction (2nd Edition)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf).
The abstract agent gives general methods, that are used by helper functions. By implementing them, agents are 
suited for experiments like those in [experiments](/experiments).
#### Dynamic Programming 
A RL approach, where model information is needed, however no exploration or random sampling. 
It will approach the true state values by using the bellman equation with the
knowledge of transition probabilities of the states and their rewards. 
#### Monte Carlo Control
As a model free approach, Monte Carlo control needs to sample data to get 
information, which the agent then can use to update his policy. To understand 
what reward can be expected for an action in a state, whole episodes get
sampled.
#### SARSA
Instead of waiting for the end of an episode, the estimations of future rewards
will be based on the estimations of the following state. 
#### N-Step TD-Control
Where MC Control uses information of the hole episode (or all steps of an episode), SARSA is only using one step
to update q-values. Generalization of both is the TD-Control Method, where you can choose the amount of future steps
that will come into account for the q-value update.  
#### Q-Learning
Table based RL-method similar to SARSA, however estimations for the following states are not based
on our actual policy of the agent (most likely epsilon greedy), but on a 
greedy target policy.
#### Double Q-Learning
Double Q-Learning is an variation of the q-learning agent where two independent q-value tables are used.
To choose an action, the sum of those is used as the single q-table from the q-learning agent.
To train the two tables, by 50% chance one of the tables will get updated, let's say the first one.
For the target action for the next state, the position in the table of the second table is used, but the value
on this position in the first table: q2[s_next, np.argmax(q1[s_next])]
instead of: np.max(self.q_table[s_next]) in the q-learning agent
#### Deep Q-Learning
This approach uses a neural network to approximate the q-table of the
Q-Learning Agent. Additionally training supporting methods like experience
replay or the use of a separate target model(similar to double q-learning)
are used. 
[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
#### Deep Q-Learning with curiosity
To improve exploration a curiosity module is used. Curiosity is defined
as reduction of uncertainty. From this follows the need of a prediction
model for future states and a evaluation metric. The worse the prediction
the more curiosity will get triggered until the agent knows the environment. 
[Curiosity-driven Exploration by Self-supervised Prediction](https://pathak22.github.io/noreward-rl)
#### Monte Carlo Tree Search
The monte carlo tree search is an approach, where whilst exploring the environment, a action tree from the starting
point is created. Promising path will get extended further, such that better action gets preferential treatment in 
ongoing training episodes.  
[Mastering the game of Go with deep neural networks and tree search](https://doi.org/10.1038%2Fnature16961)
#### Deep Deterministic Policy Gradient
Deep Deterministic Policy Gradient (DDPG) is a model-free off-policy algorithm for learning continous actions.
It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network). It uses Experience Replay and slow-learning target networks from DQN, and it is based on DPG, which can operate over continuous action spaces.
[Continuous conntrol with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf) and 
[colab research DDPG](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/ddpg_pendulum.ipynb)
