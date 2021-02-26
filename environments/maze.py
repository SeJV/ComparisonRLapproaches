from typing import List, Tuple
import copy
import math
from termcolor import colored
from gym.envs.classic_control import rendering
import numpy as np
from environments import DiscreteEnv


SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
VIEWER = rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT)

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAZE_XS = np.array([
    ['A', 'S', 'S', 'S', 'G'],
])

MAZE_SMALL = np.array([
    ['A', 'W', 'S', 'S'],
    ['S', 'S', 'S', 'W'],
    ['W', 'S', 'S', 'W'],
    ['yT', 'S', 'S', 'G'],
])

MAZE = np.array([
    ['A', 'W', 'H', 'S', 'S', 'W', 'W', 'mT'],
    ['S', 'W', 'W', 'W', 'S', 'G', 'W', 'S'],
    ['S', 'W', 'S', 'W', 'S', 'W', 'W', 'S'],
    ['S', 'S', 'S', 'W', 'S', 'S', 'S', 'S'],
    ['W', 'W', 'S', 'S', 'S', 'W', 'W', 'S'],
    ['S', 'S', 'S', 'W', 'W', 'W', 'S', 'S'],
    ['W', 'W', 'W', 'W', 'S', 'S', 'S', 'W'],
    ['yT', 'S', 'S', 'S', 'S', 'W', 'S', 'cT'],
])

MAZE_LARGE = np.array([
    ['A', 'W', 'H', 'S', 'S', 'W', 'W', 'mT', 'W', 'W', 'S', 'H'],
    ['S', 'W', 'W', 'W', 'S', 'S', 'W', 'S', 'S', 'S', 'S', 'W'],
    ['S', 'W', 'S', 'W', 'S', 'W', 'W', 'S', 'W', 'S', 'W', 'W'],
    ['S', 'S', 'S', 'W', 'S', 'S', 'S', 'S', 'W', 'S', 'S', 'S'],
    ['W', 'W', 'S', 'S', 'S', 'W', 'W', 'S', 'W', 'W', 'W', 'S'],
    ['H', 'S', 'S', 'W', 'W', 'W', 'S', 'S', 'S', 'W', 'G', 'S'],
    ['W', 'W', 'W', 'W', 'S', 'S', 'S', 'W', 'S', 'W', 'W', 'S'],
    ['S', 'S', 'S', 'S', 'S', 'W', 'S', 'W', 'S', 'S', 'S', 'S'],
    ['S', 'W', 'W', 'W', 'S', 'W', 'S', 'W', 'S', 'W', 'W', 'W'],
    ['S', 'yT', 'W', 'S', 'S', 'W', 'S', 'S', 'S', 'S', 'S', 'S'],
    ['S', 'W', 'W', 'S', 'W', 'W', 'S', 'W', 'S', 'W', 'W', 'S'],
    ['S', 'S', 'S', 'S', 'W', 'S', 'S', 'W', 'cT', 'W', 'H', 'S'],
])

MAZE_XL = np.array([
    ['A', 'W', 'S', 'W', 'W', 'W', 'S', 'S', 'S', 'S', 'S', 'S', 'W', 'S', 'S', 'S', 'S', 'S', 'W', 'mT'],
    ['S', 'W', 'S', 'W', 'S', 'W', 'S', 'W', 'W', 'W', 'W', 'S', 'S', 'S', 'W', 'W', 'W', 'S', 'W', 'S'],
    ['S', 'W', 'S', 'S', 'S', 'S', 'S', 'S', 'W', 'S', 'W', 'W', 'W', 'W', 'W', 'S', 'S', 'S', 'S', 'S'],
    ['S', 'S', 'S', 'W', 'S', 'W', 'S', 'W', 'S', 'S', 'S', 'W', 'H', 'W', 'W', 'W', 'W', 'S', 'W', 'W'],
    ['S', 'W', 'W', 'W', 'S', 'W', 'S', 'W', 'W', 'W', 'S', 'W', 'S', 'W', 'S', 'S', 'W', 'S', 'S', 'W'],
    ['S', 'W', 'S', 'S', 'S', 'W', 'S', 'S', 'S', 'W', 'S', 'W', 'S', 'S', 'S', 'S', 'W', 'W', 'S', 'S'],
    ['S', 'W', 'H', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'S', 'W', 'W', 'S', 'W', 'S', 'S', 'S', 'S', 'W'],
    ['S', 'W', 'W', 'W', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'W', 'W', 'S', 'W', 'W', 'W'],
    ['S', 'S', 'S', 'S', 'S', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'S', 'W', 'S', 'S', 'S', 'S', 'H'],
    ['W', 'W', 'S', 'W', 'W', 'W', 'S', 'S', 'S', 'S', 'W', 'H', 'W', 'S', 'W', 'W', 'W', 'S', 'W', 'W'],
    ['S', 'S', 'S', 'W', 'S', 'W', 'S', 'W', 'W', 'S', 'S', 'S', 'W', 'S', 'S', 'S', 'W', 'S', 'S', 'S'],
    ['S', 'W', 'S', 'S', 'S', 'S', 'S', 'S', 'W', 'S', 'W', 'W', 'W', 'W', 'W', 'S', 'W', 'W', 'S', 'W'],
    ['S', 'W', 'S', 'W', 'W', 'W', 'W', 'W', 'W', 'S', 'S', 'S', 'G', 'S', 'S', 'S', 'S', 'W', 'S', 'S'],
    ['H', 'W', 'S', 'S', 'S', 'W', 'S', 'S', 'S', 'S', 'W', 'W', 'W', 'W', 'S', 'W', 'W', 'W', 'W', 'S'],
    ['W', 'W', 'S', 'W', 'W', 'W', 'S', 'W', 'W', 'W', 'W', 'W', 'S', 'W', 'S', 'W', 'W', 'S', 'W', 'S'],
    ['S', 'S', 'S', 'W', 'S', 'S', 'S', 'W', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'W', 'S', 'S', 'S', 'S'],
    ['S', 'W', 'W', 'W', 'S', 'W', 'H', 'W', 'S', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'S'],
    ['S', 'S', 'S', 'W', 'S', 'W', 'W', 'W', 'S', 'W', 'S', 'S', 'S', 'S', 'S', 'S', 'W', 'S', 'S', 'S'],
    ['W', 'W', 'W', 'W', 'S', 'S', 'S', 'W', 'S', 'W', 'S', 'W', 'S', 'W', 'W', 'S', 'W', 'W', 'W', 'S'],
    ['cT', 'S', 'S', 'S', 'S', 'W', 'S', 'S', 'S', 'S', 'S', 'W', 'H', 'W', 'S', 'S', 'S', 'S', 'S', 'yT'],
])

MAZES = {
    'xs': MAZE_XS,
    's': MAZE_SMALL,
    'm': MAZE,
    'l': MAZE_LARGE,
    'xl': MAZE_XL
}


class MazeEnv(DiscreteEnv):
    def __init__(self, size: str = 'm', treasure_reward: float = 2.0, goal_hole_reward: float = 1.0):
        """
        Maze environment is a deterministic and discrete environment. The goal is to reach the goal-field, while
        collecting treasures and bypass holes. The Agent can move in four directions
        while walls and end of map restrict his movement. The goal and holes end the episode.

        :param size: There are different sizes possible for the maze, stored in MAZES global variable
        :param treasure_reward: Reward, if the agent collects a treasure
        :param goal_hole_reward: Reward scale for goal and hole, negative for the hole
        """
        self.size = size
        self.init_maze = self._choose_maze(self.size)
        self.maze = self.init_maze.copy()
        self.maze_size = np.prod(self.maze.shape)
        self.treasure_reward = treasure_reward
        self.goal_hole_reward = goal_hole_reward
        self.agent_position = self.get_agent_position()
        self.treasures_collected = {'yellow': False, 'magenta': False, 'cyan': False}

        self.mcts_moves = dict()  # for visualization of mcts moves

        # either one of the three treasures are collected or not,
        # this binary information is included in the information of the state
        nS = np.prod(self.maze.shape) * 8
        nA = 4

        # get transition probabilities
        P = dict()
        for s in range(nS):
            P[s] = {}
            P[s][LEFT] = self._calculate_transition_prob(s, LEFT)
            P[s][DOWN] = self._calculate_transition_prob(s, DOWN)
            P[s][RIGHT] = self._calculate_transition_prob(s, RIGHT)
            P[s][UP] = self._calculate_transition_prob(s, UP)

        # Calculate initial state distribution
        # We always start in state (0, 0)
        isd = np.zeros(nS)
        isd[0] = 1.0

        super(MazeEnv, self).__init__(nS, nA, P, isd)

    def reset(self) -> int:
        self.maze = self.init_maze.copy()
        self.agent_position = self.get_agent_position()
        self.treasures_collected = {'yellow': False, 'magenta': False, 'cyan': False}
        return self._get_state()

    def _new_maze(self) -> None:
        self.init_maze = self._choose_maze(self.size)

    def get_agent_position(self) -> int:
        for row_idx, row in enumerate(self.maze):
            for col_idx, col in enumerate(row):
                if col == 'A':
                    return self._coordinate_to_index(row_idx, col_idx)

    @staticmethod
    def _choose_maze(size: str) -> np.ndarray:
        return MAZES[size].copy()

    def _draw_state_on_viewer(self) -> None:
        world_width = len(self.maze[0])
        scale = SCREEN_WIDTH / world_width

        for row_idx, row in enumerate(self.maze):
            row_idx = len(self.maze) - row_idx - 1  # inverted from bottom to top
            for column_idx, field in enumerate(row):
                l, r, t, b = scale * column_idx, scale * (column_idx + 1), scale * row_idx, scale * (row_idx + 1)
                cw = (l + r) / 2
                ch = (t + b) / 2
                pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                pole.set_color(1, 1, 1)
                VIEWER.add_geom(pole)
                if field == 'W':
                    pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    pole.set_color(.2, .2, .9)
                if field == 'A':
                    pole = rendering.make_circle((scale / 2) * 0.8)
                    pole.add_attr(rendering.Transform(translation=(cw, ch)))
                    pole.set_color(.8, .2, .2)
                if field == 'yT':
                    pole = rendering.FilledPolygon([(l, ch), (cw, t), (r, ch), (cw, b)])
                    pole.set_color(0, .7, .7)
                if field == 'cT':
                    pole = rendering.FilledPolygon([(l, ch), (cw, t), (r, ch), (cw, b)])
                    pole.set_color(.7, 0, .7)
                if field == 'mT':
                    pole = rendering.FilledPolygon([(l, ch), (cw, t), (r, ch), (cw, b)])
                    pole.set_color(.7, .7, 0)
                if field == 'G':
                    pole = rendering.make_circle((scale / 2) * 0.8)
                    pole.add_attr(rendering.Transform(translation=(cw, ch)))
                    pole.set_color(.2, .9, .2)
                if field == 'H':
                    pole = rendering.make_circle((scale / 2) * 0.8)
                    pole.add_attr(rendering.Transform(translation=(cw, ch)))
                    pole.set_color(.1, .1, .1)
                VIEWER.add_geom(pole)

    def render(self, mode: str = 'human') -> None:
        self._draw_state_on_viewer()
        return VIEWER.render(return_rgb_array=mode == 'rgb_array')

    def visualize_mcts_tree(self, root_node) -> None:
        self._draw_state_on_viewer()
        for child in root_node.children:
            copy_env = copy.deepcopy(self)
            self.recursive_draw_action_tree(child, copy_env)

        max_visits = max(self.mcts_moves.values())
        for move in self.mcts_moves.keys():
            (c_cw, c_ch), (n_cw, n_ch) = move
            visits = self.mcts_moves[move]
            line_thickness = math.ceil((visits / max_visits) * 15) + 5
            line = rendering.Line((c_cw, c_ch), (n_cw, n_ch), width=line_thickness)
            line.set_color(.8, .2, .2)
            VIEWER.add_geom(line)

        return VIEWER.render()

    def recursive_draw_action_tree(self, node, env) -> None:
        world_width = len(self.maze[0])
        scale = SCREEN_WIDTH / world_width

        cp_r, cp_c = self._index_to_coordinate(env.get_agent_position())  # current point row and col cp_r, cp_c
        cp_r = len(self.maze) - cp_r - 1  # inverted from bottom to top
        env.step(node.action)
        np_r, np_c = self._index_to_coordinate(env.get_agent_position())  # new point row and col np_r, np_c
        np_r = len(self.maze) - np_r - 1  # inverted from bottom to top

        if (np_r, np_c) != (cp_r, cp_c):
            cl, cr, ct, cb = scale * cp_c, scale * (cp_c + 1), scale * cp_r, scale * (cp_r + 1)  # current point cp
            c_cw = (cl + cr) / 2   # current point center width
            c_ch = (ct + cb) / 2   # current point center height

            nl, nr, nt, nb = scale * np_c, scale * (np_c + 1), scale * np_r, scale * (np_r + 1)  # new point l, r, t, b
            n_cw = (nl + nr) / 2   # new point center width
            n_ch = (nt + nb) / 2   # new point center height

        else:  # (np_r, np_c) == (cp_r, cp_c)
            cl, cr, ct, cb = scale * cp_c, scale * (cp_c + 1), scale * cp_r, scale * (cp_r + 1)  # current point cp
            c_cw = (cl + cr) / 2   # current point center width
            c_ch = (ct + cb) / 2   # current point center height

            if node.action == LEFT:
                n_cw = c_cw - (scale / 2) * 0.9
                n_ch = c_ch
            elif node.action == DOWN:
                n_cw = c_cw
                n_ch = c_ch - (scale / 2) * 0.9
            elif node.action == RIGHT:
                n_cw = c_cw + (scale / 2) * 0.9
                n_ch = c_ch
            else:  # node.action == UP
                n_cw = c_cw
                n_ch = c_ch + (scale / 2) * 0.9

        if ((c_cw, c_ch), (n_cw, n_ch)) in self.mcts_moves.keys():
            self.mcts_moves[(c_cw, c_ch), (n_cw, n_ch)] += max(node.visits, 1)
        else:
            self.mcts_moves[(c_cw, c_ch), (n_cw, n_ch)] = max(node.visits, 1)

        for child in node.children:
            copy_env = copy.deepcopy(env)
            self.recursive_draw_action_tree(child, copy_env)

    def _move(self, action: int, row: int, col: int) -> Tuple[int, int]:
        """
        For action and position returns outcome position
        :param action: action of direction
        :param row: current row
        :param col: current column
        :return: row and column of the following state
        """
        if action == LEFT and 0 <= col - 1 < self.maze.shape[1] and self.maze[row, col - 1] != 'W':
            col = col - 1
        if action == DOWN and 0 <= row + 1 < self.maze.shape[0] and self.maze[row + 1, col] != 'W':
            row = row + 1
        if action == RIGHT and 0 <= col + 1 < self.maze.shape[1] and self.maze[row, col + 1] != 'W':
            col = col + 1
        if action == UP and 0 <= row - 1 < self.maze.shape[0] and self.maze[row - 1, col] != 'W':
            row = row - 1
        return row, col

    def step(self, action: int) -> Tuple[int, float, bool, dict]:  # -> observation, reward, done, info
        row, col = self._index_to_coordinate(self.agent_position)
        self.maze[row, col] = 'S'  # Agent gets removed from current position
        # new position: in bound, no wall
        row, col = self._move(action, row, col)

        self.agent_position = self._coordinate_to_index(row, col)
        obs = self._get_state()

        reward = 0
        done = False
        field = self.maze[row, col]
        if field in ['yT', 'mT', 'cT']:
            reward = self.treasure_reward
            if field == 'yT':
                self.treasures_collected['yellow'] = True
            if field == 'mT':
                self.treasures_collected['magenta'] = True
            if field == 'cT':
                self.treasures_collected['cyan'] = True

        self.maze[row, col] = 'A'  # agent gets added to new position

        if field == 'G':
            reward = self.goal_hole_reward
            done = True

        if field == 'H':
            reward = -self.goal_hole_reward
            done = True

        self.mcts_moves = dict()  # reset mcts moves
        return obs, reward, done, {}

    def _get_state(self) -> int:
        state = self.agent_position
        if self.treasures_collected['yellow']:
            state += self.maze_size
        if self.treasures_collected['magenta']:
            state += 2 * self.maze_size
        if self.treasures_collected['cyan']:
            state += 4 * self.maze_size

        return state

    def _index_to_coordinate(self, index: int) -> Tuple[int, int]:
        row = index // self.maze.shape[1]
        col = index % self.maze.shape[1]
        return row, col

    def _coordinate_to_index(self, row: int, col: int) -> int:
        return row * self.maze.shape[1] + col

    @staticmethod
    def _col(field: str) -> str:
        if field == 'S':
            return field
        if field == 'W':
            return colored(field, 'blue', attrs=['bold'])
        if field == 'A':
            return colored(field, 'red', attrs=['bold'])
        if field == 'yT':
            return colored('T', 'yellow', attrs=['bold'])
        if field == 'cT':
            return colored('T', 'cyan', attrs=['bold'])
        if field == 'mT':
            return colored('T', 'magenta', attrs=['bold'])
        if field == 'G':
            return colored(field, 'green', attrs=['bold'])
        if field == 'H':
            return colored('o', 'red', attrs=['bold'])

    def _calculate_transition_prob(self, s: int, action: int) -> List[Tuple[float, int, float, bool]]:
        """
        For state and action returns a list of possible state outcomes with their probability, reward and information
        if episode is done.
        :param s: state
        :param action: action
        :return: list of next_states with their probabilities, future_rewards and information if episode is done
        """
        board_index = s % self.maze_size
        row, col = self._index_to_coordinate(board_index)

        yellow_treasure_collected = False
        magenta_treasure_collected = False
        cyan_treasure_collected = False

        if (self.maze_size < s < self.maze_size * 2
                or self.maze_size * 3 < s < self.maze_size * 4
                or self.maze_size * 5 < s < self.maze_size * 6):
            yellow_treasure_collected = True
        if (self.maze_size * 2 < s < self.maze_size * 4
                or self.maze_size * 6 < s < self.maze_size * 8):
            magenta_treasure_collected = True
        if self.maze_size * 4 < s < self.maze_size * 8:
            cyan_treasure_collected = True

        row, col = self._move(action, row, col)
        state = self._coordinate_to_index(row, col)
        if yellow_treasure_collected:
            state += self.maze_size
        if magenta_treasure_collected:
            state += 2 * self.maze_size
        if cyan_treasure_collected:
            state += 4 * self.maze_size

        probability = 1.0
        reward = 0
        done = False

        if self.init_maze[row, col] == 'G':
            reward = self.goal_hole_reward
            done = True
        if self.init_maze[row, col] == 'H':
            reward = -self.goal_hole_reward
            done = True
        if self.init_maze[row, col] == 'yT' and not yellow_treasure_collected:
            state += self.maze_size
            reward = 2
        if self.init_maze[row, col] == 'mT' and not magenta_treasure_collected:
            state += 2 * self.maze_size
            reward = 2
        if self.init_maze[row, col] == 'cT' and not cyan_treasure_collected:
            state += 4 * self.maze_size
            reward = 2

        return [(probability, state, reward, done)]
