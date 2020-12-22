from typing import Optional, List
import copy
import random
from math import sqrt, log
import numpy as np
from gym import Env
from gym.envs.toy_text.discrete import DiscreteEnv
from agent_methods import AbstractAgent, DoubleQLearningAgent


BIG_NUMBER = 999999999999999999999


class _Node:
    def __init__(self, env_in_state: Env, state:  Optional[int] = None, action: Optional[int] = None,
                 reward: float = 0, done: bool = False, parent: Optional[object] = None):
        self._env_in_state = copy.deepcopy(env_in_state)  # environment with state that this node represents
        self.state = state
        self.action = action  # action which lead to this state
        self.reward = reward  # reward of state-action-next state tuple
        self.done = done  # end of episode
        self.parent: Optional[_Node] = parent  # states of time step before
        self.probability: float = 0
        self.future_rewards: float = 0
        self.visits: int = 0
        self.children: List[_Node] = []  # possible following state, action pairs

        self.a = None  # remember chosen action until train call

    @property
    def env_in_state(self):
        return copy.deepcopy(self._env_in_state)

    def get_depth(self):
        depth = 1
        depth_of_children = []
        for child in self.children:
            depth_of_children.append(child.get_depth())
        depth += max(depth_of_children) if depth_of_children else 0
        return depth


class MCTreeSearchAgent(AbstractAgent):
    """
    For stochastic discrete environments, expected values, weighted by their probability are used
    Reward function must be deterministic
    """
    def __init__(self, env: DiscreteEnv, gamma: float = 0.99, amount_test_probability: int = 1,
                 playouts_per_action: int = 10000, promising_children_playouts: int = 100, c: float = 1.41,
                 rollout_policy_agent: Optional[AbstractAgent] = None, name: str = 'MCTreeSearchAgent'):
        super().__init__(env, name=name)
        self.gamma = gamma
        # if env is stochastic, how often should an action be made, to approximate probability for next state
        self.amount_test_probability = amount_test_probability
        self.playouts_per_action = playouts_per_action  # for given state, how many playouts in total for the decision
        self.promising_children_playouts = promising_children_playouts  # how many playouts per simulation of leaf node
        self.c = c  # exploration factor of uct formula, sqrt(2) in literature, but can be changed depending on env

        self.a = None  # action which was chosen by act function
        self.root_node: Optional[_Node] = None
        self.simulation_counter = 0  # counts amount of simulation playouts

        # Agent to choose actions in simulation. Epsilon and Alpha won't get reduced, son min's are unnecessary
        self.rollout_policy_agent = rollout_policy_agent

    def reset(self) -> None:
        self.a = None  # action which was chosen by act function
        self.root_node: Optional[_Node] = None
        self.simulation_counter = 0  # counts amount of simulation playouts

    def act(self, observation: int) -> int:
        while self.simulation_counter < self.playouts_per_action:
            # 1. Selection: choose promising leaf node, that is not end of game
            if self.root_node:
                promising_leaf = self.choose_promising_leaf_node()
            else:
                promising_leaf = self.root_node = _Node(self.env)
            # 2. Expansion: expand promising node
            # test for amount_test_probability each action and count follow states, to approximate probability
            # nodes get appended with deepcopy of the environment, so rollouts can be made
            action_state_counting = dict()
            for action in range(self.env.action_space.n):
                action_state_counting[action] = dict()
                for _ in range(self.amount_test_probability):
                    env_of_leaf = promising_leaf.env_in_state
                    next_state, reward, done, _ = env_of_leaf.step(action)

                    if next_state not in action_state_counting[action].keys():
                        action_state_counting[action][next_state] = 1
                        promising_leaf.children.append(
                            # append (env, action) as child
                            _Node(env_of_leaf, state=next_state, action=action, reward=reward,
                                  done=done, parent=promising_leaf)
                        )
                    else:
                        # count occurrences of state after action
                        action_state_counting[action][env_of_leaf.s] += 1
                for child in promising_leaf.children:
                    if child.action == action:
                        child.probability = action_state_counting[action][child.state] / self.amount_test_probability
            # 3. Simulation: choose one of the new expanded nodes, simulate playouts
            rnd_child: _Node = random.choice(promising_leaf.children)
            sum_of_rewards = 0
            discount = 1
            for _ in range(self.promising_children_playouts):
                self.simulation_counter += 1
                env_simulated = rnd_child.env_in_state
                state = rnd_child.state
                done = rnd_child.done
                steps = 0
                while not done and steps < 100:
                    action = self.rollout_policy_agent.act(state)
                    state, reward, done, _ = env_simulated.step(action)
                    self.rollout_policy_agent.train(state, reward, done)

                    sum_of_rewards += discount * reward
                    discount *= self.gamma
                    steps += 1

            rnd_child.visits += self.promising_children_playouts
            rnd_child.future_rewards += sum_of_rewards
            # 4. Backpropagation: Update all parent nodes in the chain
            update_node = rnd_child
            discount = self.gamma
            while update_node.parent:
                update_node = update_node.parent
                update_node.visits += self.promising_children_playouts
                update_node.future_rewards += discount * sum_of_rewards
                discount *= self.gamma

        #  choose action with highest estimated reward
        action_list = []
        for action in range(self.env.action_space.n):
            #  get all nodes with this action and sum with their probability
            nodes_with_action = filter(lambda node: node.action == action, self.root_node.children)
            action_estimated_reward = 0
            for node in nodes_with_action:
                if node.visits:
                    action_estimated_reward += node.probability * (node.reward + (node.future_rewards / node.visits))
                elif node.done:
                    action_estimated_reward += node.probability * node.reward
            action_list.append(action_estimated_reward)

        self.a = int(np.argmax(action_list))

        print(action_list)
        self.simulation_counter = 0  # reset simulation_counter for next action decision
        self.env.render()

        return self.a

    def train(self, s_next: int, reward: float, done: bool) -> None:
        nodes_with_action_and_state = list(filter(lambda node: node.action == self.a and node.env_in_state.s == s_next,
                                                  self.root_node.children))
        if len(nodes_with_action_and_state) == 1:
            self.root_node = nodes_with_action_and_state[0]
            self.root_node.parent = None
        else:  # either state was not detected in expansion phase, or something went wrong
            self.root_node = _Node(self.env)

    def choose_promising_leaf_node(self):
        node = self.root_node
        while node.children:
            #  choose child with highest uct value
            children = sorted(node.children, key=self.uct, reverse=True)
            node = children[0]  # child with highest uct value
        return node

    def uct(self, child: _Node) -> float:
        if child.done:
            return -BIG_NUMBER
        elif child.visits == 0:
            return BIG_NUMBER
        else:
            return child.future_rewards / child.visits + self.c * sqrt(log(child.parent.visits) / child.visits)

