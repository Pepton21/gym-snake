import gym
import numpy as np
from enum import Enum
from gym import error, spaces, utils
from gym.envs.classic_control import rendering
from queue import deque
import math


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
    APPROACH = 20 / 200
    DISTANCE = -40 / 200
    SNACK = 100 / 200
    KILL = 85 / 200
    DEATH = -200 / 200


class Snake():
    DISPLACEMENT = {Direction.UP: np.array([-1, 0]), Direction.LEFT: np.array([0, -1]),
                    Direction.DOWN: np.array([1, 0]), Direction.RIGHT: np.array([0, 1])}
    LEFT_TURN = {Direction.UP: Direction.LEFT, Direction.LEFT: Direction.DOWN, Direction.DOWN: Direction.RIGHT,
                 Direction.RIGHT: Direction.UP}
    RIGHT_TURN = {Direction.UP: Direction.RIGHT, Direction.RIGHT: Direction.DOWN, Direction.DOWN: Direction.LEFT,
                  Direction.LEFT: Direction.UP}

    def __init__(self, id, start_pos, length=4, direction=Direction.RIGHT):
        self.id = id
        self.color = np.random.choice(range(256), size=3) / 256
        self.start_pos = np.array([start_pos[0], start_pos[1]])
        self.length = length
        self.direction = direction
        self.body = deque()
        self.score = 0
        for i in range(length)[::-1]:
            self.body.append(start_pos - np.array([0, i]))

    def get_head(self):
        return self.body[len(self.body) - 1]

    def move(self, ):
        self.body.append(self.get_head() + Snake.DISPLACEMENT[self.direction])

    def cut_tail(self):
        self.body.popleft()

    def inc_score(self):
        self.score += 1

    def get_score(self):
        return self.score

    def contains(self, coordinates):
        for segment in self.body:
            if np.array_equal(coordinates, segment):
                return True

    def distance_from(self, coordinates):
        head = self.get_head()
        return abs(coordinates[0] - head[0]) + abs(coordinates[1] - head[1])

    def xy_distance_from(self, coordinates):
        head = self.get_head()
        return coordinates[0] - head[0], coordinates[1] - head[1]

    def angle_from(self, coordinates):
        head = self.get_head()
        distance = math.sqrt((head[0] - coordinates[0]) ** 2 + (head[1] - coordinates[1]) ** 2)
        angle = math.acos((coordinates[0] - head[0]) / distance)
        if head[1] > coordinates[1]:
            angle = 2 * math.pi - angle
        return angle

    def nearest_point(self, points):
        min_dist = self.distance_from(points[0])
        np = points[0]
        for i in range(1, len(points)):
            dist = self.distance_from(points[i])
            if dist < min_dist:
                min_dist = dist
                np = self.distance_from(points[i])
        return np

    def check_self_collision(self):
        head = self.get_head()
        for i in range(len(self.body) - 1):
            if np.array_equal(self.body[i], head):
                return True

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
            raise Exception("Invalid action, please choose an action from [0, 1, 2]! Action chosen: {}".format(action))


class Map():
    def __init__(self, dim, obstacles=None):
        self.dim = dim
        self.grid = np.zeros(shape=dim)
        self.grid[0, :] = np.full((1, dim[0]), 1)
        self.grid[:, 0] = np.full((1, dim[0]), 1)
        self.grid[dim[0] - 1, :] = np.full((1, dim[0]), 1)
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
        free_cells = np.nonzero(grid == 0)
        idx = np.random.randint(0, len(free_cells[0]))
        grid[free_cells[0][idx], free_cells[1][idx]] = 3
        return (free_cells[0][idx], free_cells[1][idx])


class Game():
    def __init__(self, dim, snake_positions, num_snacks=1, obstacles=None):
        self.map = Map(dim, obstacles)
        self.num_snakes = len(snake_positions)
        self.num_snacks = num_snacks
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
        if len(self.snacks) < self.num_snacks:
            self.snacks.append(Map.add_snack(self.grid))

    def perform_actions(self, actions):
        info = []
        for i in range(len(actions)):
            if self.snakes[i] != None:
                result = self.snakes[i].take_action(actions[i])
                info.append("Snake {} performed {}".format(i, result))
        return info

    def remove_snack(self, x, y):
        for snack in self.snacks:
            if snack[0] == x and snack[1] == y:
                self.snacks.remove(snack)

    def right_direction(self, id, point):
        if self.snakes[id] != None:
            head = self.snakes[id].get_head()
            direction = self.snakes[id].direction
            if direction == Direction.LEFT:
                if point[1] < head[1]:
                    return 1
                else:
                    return 0
            elif direction == Direction.RIGHT:
                if point[1] > head[1]:
                    return 1
                else:
                    return 0
            elif direction == Direction.UP:
                if point[0] < head[0]:
                    return 1
                else:
                    return 0
            else:
                if point[0] > head[0]:
                    return 1
                else:
                    return 0
        return 0

    def is_point_left(self, id, point):
        if self.snakes[id] != None:
            head = self.snakes[id].get_head()
            direction = self.snakes[id].direction
            if direction == Direction.LEFT:
                if point[0] > head[0]:
                    return 1
                else:
                    return 0
            elif direction == Direction.RIGHT:
                if point[0] < head[0]:
                    return 1
                else:
                    return 0
            elif direction == Direction.UP:
                if point[1] < head[1]:
                    return 1
                else:
                    return 0
            else:
                if point[1] > head[1]:
                    return 1
                else:
                    return 0
        return 0

    def is_point_right(self, id, point):
        if self.snakes[id] != None:
            head = self.snakes[id].get_head()
            direction = self.snakes[id].direction
            if direction == Direction.LEFT:
                if point[0] < head[0]:
                    return 1
                else:
                    return 0
            elif direction == Direction.RIGHT:
                if point[0] > head[0]:
                    return 1
                else:
                    return 0
            elif direction == Direction.UP:
                if point[1] > head[1]:
                    return 1
                else:
                    return 0
            else:
                if point[1] < head[1]:
                    return 1
                else:
                    return 0
        return 0

    def get_surroundings(self, id):
        if self.snakes[id] != None:
            head = self.snakes[id].get_head()
            front_displacement = Snake.DISPLACEMENT[self.snakes[id].direction]
            left_displacement = Snake.DISPLACEMENT[Snake.LEFT_TURN[self.snakes[id].direction]]
            right_displacement = Snake.DISPLACEMENT[Snake.RIGHT_TURN[self.snakes[id].direction]]
            front = head + front_displacement
            left = head + left_displacement
            right = head + right_displacement
            front_left = head + front_displacement + left_displacement
            front_right = head + front_displacement + right_displacement
            bottom_left = head + front_displacement * (-1) + left_displacement  # right_displacement
            bottom_right = head + front_displacement * (-1) + right_displacement
            return front, front_left, left, bottom_left, front_right, right, bottom_right

    def get_integer_snake_state(self, id, discrete=True):
        if self.snakes[id] != None:
            front, front_left, left, bottom_left, front_right, right, bottom_right = self.get_surroundings(id)
            danger_front = 0
            danger_left = 0
            danger_right = 0
            if self.grid[front[0], front[1]] == 1 or self.grid[front[0], front[1]] == 2:
                danger_front = 1
            if self.grid[left[0], left[1]] == 1 or self.grid[left[0], left[1]] == 2:
                danger_left = 1
            if self.grid[right[0], right[1]] == 1 or self.grid[right[0], right[1]] == 2:
                danger_right = 1
            nearest_snack = self.snakes[id].nearest_point(self.snacks)
            right_direction = self.right_direction(id, nearest_snack)
            is_left = self.is_point_left(id, nearest_snack)
            is_right = self.is_point_right(id, nearest_snack)
            state = np.array([danger_front, danger_left, danger_right, right_direction, is_left, is_right])
            integer_state = 0
            for bit in state:
                integer_state = (integer_state << 1) | bit
            return integer_state
        else:
            return None

    def get_distances_from_nearest_snack(self):
        distances = []
        for id in self.snakes.keys():
            if self.snakes[id] is not None:
                distances.append(self.snakes[id].distance_from(self.snakes[id].nearest_point(self.snacks)))
            else:
                distances.append(0)
        return distances

    def get_snake_state(self, id, discrete=True):
        if self.snakes[id] != None:
            front, front_left, left, bottom_left, front_right, right, bottom_right = self.get_surroundings(id)
            danger_front = 0
            danger_left = 0
            danger_right = 0
            danger_front_left = 0
            danger_front_right = 0
            danger_bottom_left = 0
            danger_bottom_right = 0
            if self.grid[front_left[0], front_left[1]] == 1 or self.grid[front_left[0], front_left[1]] == 2:
                danger_front_left = 1
            if self.grid[front_right[0], front_right[1]] == 1 or self.grid[front_right[0], front_right[1]] == 2:
                danger_front_right = 1
            if self.grid[front[0], front[1]] == 1 or self.grid[front[0], front[1]] == 2:
                danger_front = 1
            if self.grid[bottom_left[0], bottom_left[1]] == 1 or self.grid[bottom_left[0], bottom_left[1]] == 2:
                danger_bottom_left = 1
            if self.grid[left[0], left[1]] == 1 or self.grid[left[0], left[1]] == 2:
                danger_left = 1
            if self.grid[bottom_right[0], bottom_right[1]] == 1 or self.grid[bottom_right[0], bottom_right[1]] == 2:
                danger_bottom_right = 1
            if self.grid[right[0], right[1]] == 1 or self.grid[right[0], right[1]] == 2:
                danger_right = 1
            nearest_snack = self.snakes[id].nearest_point(self.snacks)
            direction = self.snakes[id].direction.value
            x_distance, y_distance = self.snakes[id].xy_distance_from(nearest_snack)
            state = np.array(
                [danger_front, danger_left, danger_right, danger_front_left, danger_front_right, danger_bottom_left,
                 danger_bottom_right, x_distance / 12, y_distance / 12, direction / 3])
            return state
        else:
            return None

    def outcome(self, prior_distances):
        rewards = np.zeros((self.num_snakes,), dtype=float)
        info = []
        distances = self.get_distances_from_nearest_snack()
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
                            if self.snakes[id2] != None and self.snakes[id2].contains(head):
                                rewards[id2] += Reward.KILL.value
                                rewards[id] += Reward.DEATH.value
                                self.snakes[id] = None
                                self.dead_count += 1
                                info.append("Snake {} got killed by snake {}!".format(id, id2))
                        else:
                            if self.snakes[id] != None and self.snakes[id].check_self_collision():
                                rewards[id] += Reward.DEATH.value
                                self.snakes[id] = None
                                self.dead_count += 1
                                info.append("Snake {} killed itself!".format(id))
                if self.snakes[id] != None:
                    if self.grid[head[0], head[1]] == Cell.SNACK.value:
                        rewards[id] += Reward.SNACK.value
                        info.append("Snake {}found a snack!".format(id))
                        self.snakes[id].inc_score()
                        self.remove_snack(head[0], head[1])
                        self.snacks.append(self.map.add_snack(self.grid))
                    if rewards[id] != Reward.SNACK.value:
                        self.snakes[id].cut_tail()
                    if distances[id] < prior_distances[id]:
                        rewards[id] += Reward.APPROACH.value
                    else:
                        rewards[id] += Reward.DISTANCE.value
        return rewards, info

    def step(self, actions):
        info = {}
        prior_distances = self.get_distances_from_nearest_snack()
        actions = self.perform_actions(actions)
        for action in actions:
            info[len(info)] = action
        reward, events = self.outcome(prior_distances)
        done = self.dead_count == self.num_snakes
        for event in events:
            info[len(info)] = event
        self.update_grid()
        observation = self.grid
        return observation, reward, done, info


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_dim=(15, 15), snake_positions=[(7, 7)]):
        self.map_dim = map_dim
        self.snake_positions = snake_positions
        self.game = Game(dim=map_dim, snake_positions=snake_positions)
        self.action_space = spaces.Box(0, 2, shape=(self.game.num_snakes,), dtype=np.int16)
        self.observation_space = spaces.Box(0, 1, shape=(10,), dtype=np.int16)
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
