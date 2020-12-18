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
Functions to train agents in environments, which results in stats for visualisation or
other analysis. 

#### train agents
This function you can give an (openAI) environment and a list of agents, which are instances of
your rl-approaches. It will train every agent for a certain amount of episodes on the environment.
Those trainings will get repeated multiple times for every agent. The results can be represented 
by the utils function _visualize_training_results_for_agents_

#### train agent
This is a simpler function to train one agent a single time for a certain amount of episodes. 
**train_agents** is recommended for most use cases, which makes use of this function. 

### agents
Agent classes are implementations for some methods found in literature for approaches to solve
reinforcement learning problems. Most of them can be found in [Reinforcement Learning - An
Introduction by Richard S. Sutton and Andrew G. Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf). 
#### Dynamic Programming 
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


 
