## Comparison of Reinforcement Learning approaches
In this repository reinforcement learning approaches can be tested and compared with
environments of OpenAI and others.

This is an handy tool, to compare results of different approaches or the same approach with
different hyperparameters on environments.  

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
reinforcement learning problems. Most of them can be found in [Reinforcement Learning - An
Introduction by Richard S. Sutton and Andrew G. Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf). 
#### Dynamic Programming 
A RL approach, where model information is needed, however no exploration or random sampling. 
It will approach the true state values by using the bellman equation with 
#### off-policy MC Control
DEBUG
#### n-step TD-Control
TODO
#### SARSA
#### Q-Learning
#### Double Q-Learning
#### MC-Tree Search
TODO
#### Deep Q-Learning
#### Deep Q-Learning with curiosity


 
