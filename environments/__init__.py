# abstract discrete env
from gym.envs.toy_text.discrete import DiscreteEnv

# Discrete observation and action space:
from environments.maze import MazeEnv
from gym.envs.toy_text.cliffwalking import CliffWalkingEnv
from gym.envs.toy_text.taxi import TaxiEnv
# can be init with map_size='8x8' or create random with desc = generate_random_map(size) and FrozenLakeEnv(desc)
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

# Continuous observation space and discrete action space:
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.classic_control.mountain_car import MountainCarEnv
from gym.envs.classic_control.acrobot import AcrobotEnv  # pendulum with two parts an discrete action

# Continuous observation and action space:
from gym.envs.classic_control.pendulum import PendulumEnv
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

# Atari Envs import with gym.make('id') all with continuous observation space and discrete action space

# Constants, for which avg reward the env is considered solved
FROZEN_LAKE_SOLVED_AT = 0.78
CART_POLE_SOLVED_AT = 475.0
