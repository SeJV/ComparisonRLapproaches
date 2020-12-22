# Comparison of Reinforcement Learning approaches
In this repository reinforcement learning approaches can be tested and compared with
environments of OpenAI and others.

This is an handy tool, to compare results of different approaches or the same approach with
different hyperparameters on environments.  


## How to use it
Creating an environment with `gym.make(<gymid>)` or by importing environment from `/environnments` where all the working
envs are imported. 

```python
from environments import FrozenLakeEnv
env = FrozenLakeEnv(map_name='8x8')
```

To test some reinforcement learning methods, you can either implement your own method and comply with the abstract_agent
from `rl_methods`. With their class implementation it is also described, for what type of environment they should
work. 

```python
from rl_methods import SarsaAgent, QLearningAgent
sarsa_agent = SarsaAgent(env)
q_learning_agent = QLearningAgent(env)
``` 

To train those agents and to visualise the statistics that emerged during training,
functions from `/train` and `/utils` are used. 

```python
from train import train_agents
from utils import visualize_training_results_for_agents

stats = train_agents(env, [sarsa_agent, q_learning_agent])
visualize_training_results_for_agents(stats)
```

Examples of usage can be found in `/examples`


### Environments
Environments can be loaded from the [open ai package gym](https://gym.openai.com) or 
implemented as an gym environment subclass.
#### maze
This is an additionally implemented discrete deterministic environment which especially can be 
used to compare discovery behavior of unseen states. Here good exploration is helpful to find
all treasures and get to the goal.   

### Train function
Functions to train agents in environments, which results in stats for visualisation and further
analysis.  

#### train_agents
This function you can pass an env and a list of agents, which are instances of rl-approaches.
It will train every agent for a certain amount of episodes on the environment. Those trainings
will get repeated multiple times for every agent. The results can be represented by the utils
function _visualize_training_results_for_agents_.

#### train_agent
This is a basis function to train one agent once for a certain amount of episodes. 
**train_agents** is recommended for most use cases, which makes use of this function. 

### Agents
Agent classes are implementations for some methods found in literature for approaches to solve
reinforcement learning problems. Methods without further sources can be found in [Reinforcement Learning - An
Introduction by Richard S. Sutton and Andrew G. Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf). 
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
#### n-step TD-Control
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
#### MC-Tree Search
The monte carlo tree search is an approach, where whilst exploring the environment, a action tree from the starting
point is created. Promising path will get extended further, such that better action gets preferential treatment in 
ongoing training episodes.  
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


 
