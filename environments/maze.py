from termcolor import colored
from gym.envs.toy_text import discrete
import numpy as np

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

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
    's': MAZE_SMALL,
    'm': MAZE,
    'l': MAZE_LARGE,
    'xl': MAZE_XL
}


class MazeEnv(discrete.DiscreteEnv):
    def __init__(self, size='m', treasure_reward=1.0, goal_hole_reward=1.0):
        self.size = size
        self.init_maze = self._choose_maze(self.size)
        self.maze = self.init_maze.copy()
        self.maze_size = np.prod(self.maze.shape)
        self.treasure_reward = treasure_reward
        self.goal_hole_reward = goal_hole_reward
        self.agent_position = self._get_agent_position()
        self.treasures_collected = {'yellow': False, 'magenta': False, 'cyan': False}

        # either three are treasures collected or not, so times 2x2x2
        nS = np.prod(self.maze.shape) * 8
        nA = 4

        # TODO: get transition probabilities
        P = {}
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

    def reset(self):
        self.maze = self.init_maze.copy()
        self.agent_position = self._get_agent_position()
        self.treasures_collected = {'yellow': False, 'magenta': False, 'cyan': False}
        return self._get_state()

    def _new_maze(self):
        self.init_maze = self._choose_maze(self.size)

    def _get_agent_position(self):
        for row_idx, row in enumerate(self.maze):
            for col_idx, col in enumerate(row):
                if col == 'A':
                    return self._coordinate_to_index(row_idx, col_idx)

    @staticmethod
    def _choose_maze(size):
        return MAZES[size].copy()

    def render(self, mode='human'):
        row, col = self._index_to_coordinate(self.agent_position)
        maze_copy = self.maze.copy()
        maze_copy[row, col] = 'A'
        for row in maze_copy:
            print(*[self._col(field) for field in row])
        print('\n')

    def _move(self, action, row, col):
        if action == LEFT and 0 <= col - 1 < self.maze.shape[1] and self.maze[row, col - 1] != 'W':
            col = col - 1
        if action == DOWN and 0 <= row + 1 < self.maze.shape[0] and self.maze[row + 1, col] != 'W':
            row = row + 1
        if action == RIGHT and 0 <= col + 1 < self.maze.shape[1] and self.maze[row, col + 1] != 'W':
            col = col + 1
        if action == UP and 0 <= row - 1 < self.maze.shape[0] and self.maze[row - 1, col] != 'W':
            row = row - 1
        return row, col

    def step(self, action):  # -> observation, reward, done, info
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

        return obs, reward, done, {}

    def _get_state(self):
        state = self.agent_position
        if self.treasures_collected['yellow']:
            state += self.maze_size
        if self.treasures_collected['magenta']:
            state += 2 * self.maze_size
        if self.treasures_collected['cyan']:
            state += 4 * self.maze_size

        return state

    def _index_to_coordinate(self, index):
        row = index // self.maze.shape[1]
        col = index % self.maze.shape[1]
        return row, col

    def _coordinate_to_index(self, row, col):
        return row * self.maze.shape[1] + col

    @staticmethod
    def _col(field):
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

    def _calculate_transition_prob(self, s, action):
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
