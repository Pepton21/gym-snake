import gym
import numpy as np
from enum import Enum
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering
from queue import deque


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Action(Enum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2

class Cell(Enum):
    FREE = 0
    WALL = 1
    SNAKE = 2
    SNACK = 3

class Reward(Enum):
    SNACK = 10
    KILL = 50
    DEATH = -50

class Snake():
    DISPLACEMENT = {Direction.UP: np.array([-1, 0]), Direction.LEFT: np.array([0, -1]),
                    Direction.DOWN: np.array([1, 0]), Direction.RIGHT: np.array([0, 1])}
    LEFT_TURN = {Direction.UP: Direction.LEFT, Direction.LEFT: Direction.DOWN, Direction.DOWN: Direction.RIGHT,
                 Direction.RIGHT: Direction.UP}
    RIGHT_TURN = {Direction.UP: Direction.RIGHT, Direction.RIGHT: Direction.DOWN, Direction.DOWN: Direction.LEFT,
                  Direction.LEFT: Direction.UP}

    def __init__(self, id, start_pos, length=4, direction = Direction.RIGHT):
        self.id = id
        self.color = np.random.choice(range(256), size=3)/256
        self.start_pos = np.array([start_pos[0], start_pos[1]])
        self.length = length
        self.direction = direction
        self.body = deque()
        for i in range(length)[::-1]:
            self.body.append(start_pos - np.array([0, i]))

    def get_head(self):
        return self.body[len(self.body)-1]

    def move(self, ):
        self.body.append(self.get_head()+Snake.DISPLACEMENT[self.direction])
        #if grow == False:
        #    self.body.popleft()
    def cut_tail(self):
        self.body.popleft()

    def take_action(self, action):
        if action == Action.STRAIGHT.value:
            self.move()
            return Action.STRAIGHT
        elif action == Action.LEFT.value:
            self.direction = Snake.LEFT_TURN[self.direction]
            self.move()
            return Action.LEFT
        elif action == Action.RIGHT.value:
            self.direction = Snake.RIGHT_TURN[self.direction]
            self.move()
            return Action.RIGHT
        else:
            raise Exception("Invalid action, please choose an action from [0, 1, 2]!")

class Map():
    def __init__(self, dim, obstacles=None):
        self.dim = dim
        self.grid = np.zeros(shape=dim)
        self.grid[0, :] = np.full((1, dim[0]), 1)
        self.grid[:, 0] = np.full((1, dim[0]), 1)
        self.grid[dim[0]-1, :] = np.full((1, dim[0]), 1)
        self.grid[:, dim[0] - 1] = np.full((1, dim[0]), 1)

        if obstacles != None:
            for obstacle in obstacles:
                self.grid[obstacle[0], obstacle[1]] = 1

    def get_grid(self):
        return self.grid

    @staticmethod
    def add_snake(grid, snake):
        for segment in snake.body:
            grid[segment[0], segment[1]] = 2

    @staticmethod
    def add_snack(grid):
        free_cells = np.nonzero(grid==0)
        idx = np.random.randint(0, len(free_cells[0]))
        grid[free_cells[0][idx], free_cells[1][idx]] = 3

class Game():
    def __init__(self, dim, snake_positions, obstacles=None):
        self.map = Map(dim, obstacles)
        self.num_snakes = len(snake_positions)
        self.snakes = {}
        self.dead_count = 0
        self.snacks = []
        for snake_pos in snake_positions:
            id = len(self.snakes)
            self.snakes[id] = Snake(id=id, start_pos=snake_pos)
        self.update_grid()

    def add_snake(self, snake_pos):
        id = len(self.snakes)
        self.snakes[id] = Snake(id=id, start_pos=snake_pos)
        self.num_snakes += 1
        self.update_grid()

    def update_grid(self):
        if hasattr(self, 'grid'):
            self.grid[self.grid == 2] = 0
        else:
            self.grid = np.copy(self.map.get_grid())

        for key in self.snakes.keys():
            if self.snakes[key] != None:
                Map.add_snake(self.grid, self.snakes[key])
        if (Cell.SNACK.value in self.grid) == False:
            Map.add_snack(self.grid)

    def perform_actions(self, actions):
        info = []
        for i in range(len(actions)):
            if self.snakes[i] != None:
                result = self.snakes[i].take_action(actions[i])
                info.append("Snake {} performed {}".format(i, result))
        return info

    def outcome(self):
        rewards = np.zeros((self.num_snakes,), dtype=int)
        info = []
        for id in self.snakes.keys():
            if self.snakes[id] != None:
                head = self.snakes[id].get_head()
                if self.grid[head[0], head[1]] == Cell.WALL.value:
                    rewards[id] += Reward.DEATH.value
                    self.snakes[id] = None
                    self.dead_count += 1
                    info.append("Snake {} hit a wall!".format(id))
                else:
                    for id2 in self.snakes.keys():
                        if id2 != id:
                            if self.snakes[id2] != None:
                                for i in range(len(self.snakes[id2].body)):
                                    body = self.snakes[id2].body
                                    if np.array_equal(body[i], head):
                                        rewards[id2] += Reward.KILL.value
                                        rewards[id] += Reward.DEATH.value
                                        self.snakes[id] = None
                                        self.dead_count += 1
                                        info.append("Snake {} got killed by snake {}!".format(id, id2))
                        else:
                            if self.snakes[id] != None:
                                for i in range(len(self.snakes[id].body)-1):
                                    body = self.snakes[id].body
                                    if np.array_equal(body[i], head):
                                        rewards[id] += Reward.DEATH.value
                                        self.snakes[id] = None
                                        self.dead_count += 1
                                        info.append("Snake {} killed itself!".format(id))
                                        break

                if self.grid[head[0], head[1]] == Cell.SNACK.value:
                    rewards[id] += Reward.SNACK.value
                    info.append("Snake {}found a snack!".format(id))
                if rewards[id] != Reward.SNACK.value:
                    if self.snakes[id] != None:
                        self.snakes[id].cut_tail()
        return rewards, info

    def step(self, actions):
        info = {}
        actions = self.perform_actions(actions)
        for action in actions:
            info[len(info)] = action
        reward, events = self.outcome()
        print(self.dead_count, self.num_snakes)
        done = self.dead_count == self.num_snakes
        for event in events:
            info[len(info)] = event
        self.update_grid()
        observation = self.grid
        return observation, reward, done, info




class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_dim=(15,15), snake_positions=[(7,7)]):
        self.map_dim = map_dim
        self.snake_positions = snake_positions
        self.game = Game(dim=map_dim, snake_positions=snake_positions)
        self.action_space = spaces.Box(0, 2, shape=(self.game.num_snakes,), dtype=np.int16)
        self.observation_space = spaces.Box(0, 3, shape=map_dim, dtype=np.int16)
        self.viewer = None


    def add_snake(self, snake_pos):
        self.snake_positions.append(snake_pos)
        self.game.add_snake(snake_pos)
        self.action_space = spaces.Box(0, 2, shape=(self.game.num_snakes,), dtype=np.int16)

    def step(self, action):
        return self.game.step(action)

    def reset(self):
        self.game = Game(dim=self.map_dim, snake_positions=self.snake_positions)
        return self.game.grid

    def render(self, mode='human'):
        #print(self.game.grid)
        width = height = 600
        width_scaling_factor = width / self.map_dim[0]
        height_scaling_factor = height / self.map_dim[1]

        if self.viewer is None:
            self.viewer = rendering.Viewer(width, height)

        wall_cells = np.nonzero(self.game.grid == 1)
        for x, y in zip(wall_cells[0], wall_cells[1]):
            l, r, t, b = x * width_scaling_factor, (x + 1) * width_scaling_factor, y * height_scaling_factor, (
                        y + 1) * height_scaling_factor
            square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            square.set_color(0, 0, 0)
            self.viewer.add_onetime(square)

        snack_cells = np.nonzero(self.game.grid == 3)
        for x, y in zip(snack_cells[0], snack_cells[1]):
            l, r, t, b = x * width_scaling_factor, (x + 1) * width_scaling_factor, y * height_scaling_factor, (
                        y + 1) * height_scaling_factor
            square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            square.set_color(0.75, 0.25, 0.25)
            self.viewer.add_onetime(square)

        for key in self.game.snakes.keys():
            snake = self.game.snakes[key]
            if snake != None:
                for segment in snake.body:
                    x = segment[0]
                    y = segment[1]
                    l, r, t, b = x * width_scaling_factor, (x + 1) * width_scaling_factor, y * height_scaling_factor, (
                            y + 1) * height_scaling_factor
                    square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    square.set_color(snake.color[0], snake.color[1], snake.color[2])
                    self.viewer.add_onetime(square)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

"""snake = Snake(1, (5,5))
print(snake.body)
map = Map((15,15))
print(map.get_grid())
Map.add_snake(map.get_grid(), snake)
print(map.get_grid())
print(3 in map.get_grid())
Map.add_snack(map.get_grid())
print(map.get_grid())
print(3 in map.get_grid())
a = spaces.Box(0, 2, shape=(3,), dtype=np.int16)
print(a.sample())"""
