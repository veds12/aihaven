import gym
import numpy as np

from q_learning import agent

STEP_SIZE = 0.1
GAMMA = 0.9
EPSILON = 0.1

env = gym.make("FrozenLake-v0")

myagent = agent(STEP_SIZE, GAMMA, EPSILON, env)
myagent.train()
myagent.display_value()
