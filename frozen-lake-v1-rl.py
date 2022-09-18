from venv import create
import numpy as np
import gym
import random
import time
from IPython.display import clear_output
import pygame as pygame

# get enviorment from gym library
env = gym.make("FrozenLake-v1", render_mode="human")


# get size of the enviorment from gym library
action_space_size = env.action_space.n
state_space_size = env.observation_space.n


# build a q table for the envirment
q_table = np.zeros((state_space_size, action_space_size))


# define variable

episodes = 10000
steps = 100

learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01


all_episodes_reward = []

pygame.init()

# q learning alogorithm

# for each episode
for episode in range(episodes):
    state, info = env.reset()
    terminated = False
    truncated = False

    current_reward = 0

    # for each step in in each episode
    for step in range(steps):

        # get random number from as threshold for exploration and exploitation
        exploration_threshhold = random.uniform(0, 1)

        if exploration_threshhold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        env.render()

        # update q table with new values
        q_table[state, action] = q_table[state, action] * (
            1 - learning_rate
        ) + learning_rate * (reward + discount_rate * np.max(q_table[observation, :]))

        print(reward, exploration_rate, q_table)

        state = observation
        current_reward += reward

        if terminated or truncated:
            break

    exploration_rate = min_exploration_rate + (
        max_exploration_rate - min_exploration_rate
    ) * np.exp(-exploration_decay_rate * episode)

    all_episodes_reward.append(current_reward)


print(q_table)

env.close()
