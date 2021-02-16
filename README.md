# Comparison of Reinforcement Learning approaches
In this repository reinforcement learning approaches can be tested and compared with
environments of OpenAI and others.

This is an handy tool, to compare results of different approaches or the same approach with
different hyperparameters on environments.  


## How to use it
Examples of usage can be found in `/experiments` as well as an explanation for
those here: [experiments](/experiments).

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
from agents import SarsaAgent, QLearningAgent
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
Details for Agent implementations can be found here: [agents](/agents).

 
