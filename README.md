# Comparison of multiple Reinforcement Learning approaches
In this repository many basic Reinforcement Learning approaches are can be tested with environments of OpenAI and others.

This is an handy tool, to compare results of different approaches an hyperparameters on multiple environments. 

## Tests
Functions to train agents in environments, which results in stats for visualisation or other analysis. 
### train agents
This function you can give an (openAI) environment and a list of agents, which are instances of
your rl-approaches. It will train every agent for a certain amount of episodes on the environment. Those
trainings will get repeated multiple times for every agent. The results can be represented by the utils function
_visualize_training_results_for_agents_

### train agent
This is a simpler function to train one agent a single time for a certain amount of episodes. 
**train_agents** is recommended for most use cases, which makes use of this function. 

## agents
Agent classes are implementations for some methods found in literature for approaches to solve
reinforcement learning problems. Most of them can be found in [Link](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf). 
### Dynamic Programming
### off-policy MC Control
### SARSA
### Q-Learning
### Double Q-Learning
### Double Q-Learning with curiosity
### Deep Q-Learning

## environments
Here are the environments on which the agents can be tested. They can also be tested on other discrete, or 
discretisized environments in the openAI schema of gyms. 
### cliffwalking
### maze

 