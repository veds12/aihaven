import math
import random
from collections import deque, namedtuple

import gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class agent:
    def __init__(
        self,
        env,
        GAMMA,
        EPSILON=0,
        hidden_layers_size=[24, 24],
        buffer_size=10000,
    ):
        self.env = env
        self.dqn = self.model(hidden_layers_size)
        self.memory = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.GAMMA = GAMMA
        self.EPSILON = EPSILON

    def model(self, hidden_layers_size):
        return (
            nn.Sequential(
                nn.Linear(
                    self.env.observation_space.shape[0],
                    hidden_layers_size[0],
                ),
                nn.ReLU(),
                nn.Linear(hidden_layers_size[0], hidden_layers_size[1]),
                nn.ReLU(),
                nn.Linear(hidden_layers_size[1], self.env.action_space.n),
            )
            .to(device)
            .to(dtype)
        )

    def select_action(self, state):
        if random.uniform(0, 1) < self.EPSILON:
            return self.env.action_space.sample()
        else:
            return torch.argmax(self.dqn(state))

    def store_transitions(self, state, action, reward, next_state, done):
        if len(self.memory) < self.buffer_size:
            self.memory.append(
                Transition(state, action, reward, next_state, done)
            )
        else:
            self.memory.popleft()
            self.memory.append(
                Transition(state, action, reward, next_state, done)
            )

    def sample(self, batch_size):
        return Transition(
            *[
                torch.cat[i]
                for i in [
                    *zip(
                        *random.sample(
                            self.memory, min(len(self.memory), batch_size)
                        )
                    )
                ]
            ]
        )

    def play(self, EPISODES, learning_rate=0.01):
        for i in range(EPISODES):
            state = self.env.reset()
            for j in range(5000):
                self.env.render()
                action = self.select_action(torch.from_numpy(state))
                next_state, reward, done, _ = self.env.step(action)
                self.store_transitions(
                    state, action, reward, next_state, done
                )
                sample = self.sample(32)
                if sample.done:
                    target = sample.reward
                else:
                    target = sample.reward + self.GAMMA * torch.max(
                        self.dqn(sample.next_state)
                    )
                curr_pred = self.dqn(sample.state)[sample.action]
                optimizer = optim.nn.Adam(
                    self.dqn.parameters(), lr=learning_rate
                )
                loss = nn.MSELoss()(target, curr_pred)
                print(loss.item())
                loss.backward()
                optimizer.step()
                state = next_state
                if done:
                    break
        self.env.close()

    def test(self, episodes):
        self.env.render()
        for e in range(episodes):
            state = self.env.reset()
            done = False
            i = 0
            while not done:
                action = torch.argmax(self.dqn(state))
                state, reward, done, _ = self.env.step(action)
                i += 1
                if done:
                    print(
                        "Episode no. : {}/{} , Score : {}".format(
                            e, episodes, i
                        )
                    )


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    GAMMA = 0.9
    EPSILON = 0.2
    hidden_layers_size = [24, 24]
    buffer_size = 10000
    batch_size = 32
    EPISODES = 1000
    episodes = 20
    lr = 0.05

    myagent = agent(env, GAMMA, EPSILON, hidden_layers_size, buffer_size)
    myagent.play(EPISODES, lr)
    myagent.test(episodes)
