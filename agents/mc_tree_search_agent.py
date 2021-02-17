from typing import Optional, List
import copy
import random
from math import sqrt, log
import numpy as np
from environments import DiscreteEnv, MazeEnv
from agents import AbstractAgent


BIG_NUMBER = 999999999999999999999


class Node:
    def __init__(self, action: Optional[int] = None, parent: Optional[object] = None):
        self.action = action  # action taken
        self.parent: Optional[Node] = parent  # states of time step before
        self.future_rewards: float = 0
        self.reward: float = 0
        self.visits: int = 0
        self.children: List[Node] = []  # possible following state, action pairs

    def get_action_chain(self) -> List[int]:
        """
        :return: list of actions, from root node, to this action-node
        """
        action_chain = [self.action] if self.action else []
        node_to_add_action = self.parent
        while node_to_add_action and node_to_add_action.parent:
            action_chain.append(node_to_add_action.action)
            node_to_add_action = node_to_add_action.parent

        return list(reversed(action_chain))

    def get_depth(self) -> int:
        """
        :return: depth of node from root
        """
        depth = 1
        depth_of_children = []
        for child in self.children:
            depth_of_children.append(child.get_depth())
        depth += max(depth_of_children) if depth_of_children else 0
        return depth

    def generate_tree(self) -> dict:
        """
        for easier debugging, generate tree as json
        """
        root_dict = {}
        for child in self.children:
            root_dict[child.action] = {
                'visits': child.visits,
                'future_rewards': child.future_rewards,
                'children': child.generate_tree()
            }
        return root_dict


class MCTreeSearchAgent(AbstractAgent):
    def __init__(self, env: DiscreteEnv, alpha: float = 0.01, alpha_min: float = 0,
                 alpha_reduction: float = 0.0, gamma: float = 0.99, playouts_per_action: int = 10000,
                 promising_children_playouts: int = 100, c: float = 1.41,
                 rollout_policy_agent: Optional[AbstractAgent] = None, visualize: bool = False,
                 name: str = 'MCTreeSearchAgent'):
        super().__init__(env, alpha=alpha, alpha_min=alpha_min, alpha_reduction=alpha_reduction, name=name)
        self.gamma = gamma

        self.playouts_per_action = playouts_per_action  # for given state, how many playouts in total for the decision
        self.promising_children_playouts = promising_children_playouts  # how many playouts per simulation of leaf node
        self.c = c  # exploration factor of uct formula, sqrt(2) in literature, but can be changed depending on env

        self.a = None  # action which was chosen by act function
        self.root_node: Optional[Node] = None
        self.simulation_counter = 0  # counts amount of simulation playouts

        self.visualize = visualize  # shows path of nodes, if env is supported

        # Agent to choose actions in simulation. Epsilon and Alpha won't get reduced, son min's are unnecessary
        self.rollout_policy_agent = rollout_policy_agent

    def reset(self) -> None:
        self.a = None  # action which was chosen by act function
        self.root_node: Optional[Node] = None
        self.simulation_counter = 0  # counts amount of simulation playouts

    def get_possible_actions(self, node: Node) -> range:
        # return actions possible for env, probably depending on node
        return range(self.env.action_space.n)

    def act(self, observation: int) -> int:
        while self.simulation_counter < self.playouts_per_action:
            # 1. Selection: choose promising leaf node, that is not end of game
            if self.root_node:
                promising_leaf = self.choose_promising_leaf_node()
            else:
                promising_leaf = self.root_node = Node()
            # 2. Expansion: expand promising node
            for action in self.get_possible_actions(promising_leaf):
                promising_leaf.children.append(Node(action=action, parent=promising_leaf))
            # 3. Simulation: choose one of the new expanded nodes, simulate playouts
            actions_to_promising = promising_leaf.get_action_chain()

            for _ in range(self.promising_children_playouts):
                # start from root node, execute action until the node
                root_env_copy = copy.deepcopy(self.env)
                is_done = False
                for action in actions_to_promising:
                    _, _, done, _ = root_env_copy.step(action)
                    is_done = is_done or done
                self.simulation_counter += 1

                if not is_done:
                    for child in promising_leaf.children:
                        root_env_copy_for_child = copy.deepcopy(root_env_copy)
                        discount = self.gamma

                        state, reward, done, _ = root_env_copy_for_child.step(child.action)
                        if child.reward != 0:
                            child.reward = (1 - self.alpha) * child.reward + self.alpha * reward
                        else:
                            child.reward = reward
                        sum_of_future_rewards = 0
                        steps = 0
                        while not done and steps < 100:
                            action = self.rollout_policy_agent.act(state)
                            state, reward, done, _ = root_env_copy_for_child.step(action)
                            # self.rollout_policy_agent.train(state, reward, done)

                            sum_of_future_rewards += discount * reward
                            discount *= self.gamma
                            steps += 1

                        child.visits += 1
                        if child.future_rewards != 0:
                            child.future_rewards = (
                                (1 - self.alpha) * child.future_rewards + self.alpha * sum_of_future_rewards)
                        else:
                            child.future_rewards = sum_of_future_rewards
                    # 4. Backpropagation: Update all parent nodes in the chain
                    update_node = child
                    while update_node.parent:
                        update_node = update_node.parent
                        update_node.visits += 1

            if self.visualize and isinstance(self.env, MazeEnv):
                self.env.visualize_mcts_tree(self.root_node)

        #  choose action with highest estimated reward
        children_values = self._get_child_values()
        self.a = int(np.argmax(children_values))
        self.simulation_counter = 0  # reset simulation_counter for next action decision
        return self.a

    def train(self, s_next: int, reward: float, done: bool) -> None:
        if self.root_node.children:
            for child in self.root_node.children:
                if child.action == self.a:
                    self.root_node = child
            self.root_node.parent = None
        else:  # either state was not detected in expansion phase, or something went wrong
            self.root_node = Node()

    def choose_promising_leaf_node(self) -> Node:
        node = self.root_node
        while node.children:
            #  choose node with highest uct value
            children = sorted(node.children, key=self.uct, reverse=True)
            node = children[0]  # node with highest uct value
        return node

    def uct(self, node: Node) -> float:
        if node.visits == 0:
            return BIG_NUMBER
        else:
            return self._get_node_value(node) + self.c * sqrt(log(node.parent.visits) / node.visits)

    def _get_child_values(self) -> List[float]:
        child_values = []
        for child in self.root_node.children:
            child_values.append(self._get_node_value(child))
        return child_values

    def _get_node_value(self, node: Node) -> float:
        if not node.children:
            return node.reward + node.future_rewards
        else:
            child_values = list([self._get_node_value(child) for child in node.children])
            return node.reward + self.gamma * max(child_values)
