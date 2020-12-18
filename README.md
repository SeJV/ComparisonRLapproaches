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
_Open to do_
#### Q-Learning
Similar to SARSA, however estimations for the following states are not based
on our actual policy of the agent (most likely epsilon greedy), but on a 
greedy target policy.
#### Double Q-Learning
We use two separate q-tables, where the sum decides the chosen action. 
We will update randomly one at a time, where estimations for the 
future states are supported by the respective other table. 
#### MC-Tree Search
_Open to do_
#### Deep Q-Learning
This approach uses a neural network to approximate the q-table of the
Q-Learning Agent. Additionally training supporting methods like experience
replay or the use of a separate target model(similar to double q-learning)
are used. 

#### Deep Q-Learning with curiosity
To improve exploration a curiosity module is used. Curiosity is defined
as reduction of uncertainty. From this follows the need of a prediction
model for future states and a evaluation metric. The worse the prediction
the more curiosity will get triggered until the agent knows the environment. 


 
