# Overview
In this repository some of the most popular reinforcement learning approaches are implemented. 
They all follow the structure of one [abstract agent](agents/abstract_agent.py), in such a way, that helper functions 
to compare agents can be used on all instances. These are meant to serve as a learning and comparison tool. 

## Table of Contents
- [Agents](/agents)
- [Environments](/environments)
- [Experiments](/experiments)
- Helper Functions in: [utils](/utils) and [train](/train)

### How to use it
Examples of usage can be found in [/experiments](/experiments) as well as an explanation for those. 

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

### Resources
- [Reinforcement Learning - An Introduction (2nd Edition)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [Curiosity-driven Exploration by Self-supervised Prediction](https://pathak22.github.io/noreward-rl)
- [Mastering the game of Go with deep neural networks and tree search](https://doi.org/10.1038%2Fnature16961)
- [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)
- [colab research DDPG](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/rl/ipynb/ddpg_pendulum.ipynb)

 
