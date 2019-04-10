import gym
from queue import deque
from gym import spaces
import numpy as np
import time
from gym_snake.envs import SnakeEnv

#env = gym.make('CartPole-v0')
env = SnakeEnv()
env.add_snake((5,7))
env.add_snake((9,7))
for i_episode in range(100):
    observation = env.reset()
    for t in range(100):
        env.render()
        time.sleep(0.2)
        print(observation)

        action = env.action_space.sample()
        print('action:',action)
        observation, reward, done, info = env.step(action)
        print('reward:', reward)
        print(info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

