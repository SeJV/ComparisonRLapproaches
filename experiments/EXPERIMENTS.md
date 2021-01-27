# Experiments

In general, all experiments follow the same structure: 

```python
from environments import DiscreteEnv
from agents import AbstractAgent
from train import train_agents
from utils import visualize_training_results_for_agents

# initialize environment
env = DiscreteEnv()

# create different agents, you want to train
agent1 = AbstractAgent(env)
agent2 = AbstractAgent(env)
agent3 = AbstractAgent(env)

# train list of agents on the same environment
stats = train_agents(env, [agent1, agent2, agent3])
# store the stats as a plot
visualize_training_results_for_agents(stats)
```

## Comparison

### Table based methods
<img src="./plots/table_based_models_frozen_lake.png" width="400" />

### NTD prediction
<img src="./plots/comparison_n_step_td_prediction.png" width="400" />

### DQN hyperparameters
<img src="./plots/comparison_dqn_cart_pole.png" width="400" />

## Showcase

### Deep deterministic policy gradient agent
<img src="./plots/ddpg_on_pendulum.png" width="400" />

### Monte carlo tree search
<img src="./plots/mcts_on_medium_maze.png" width="400" />
  
