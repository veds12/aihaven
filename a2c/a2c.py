# Author : Vedant Shah
# E-mail : vedantshah2012@gmail.com

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class agent:
    def __init__(
        self, env, actor_layer_sizes=[64, 32], critic_layer_sizes=[64, 32]
    ):
        self.env = env
        self.actor, self.critic = self.make_models(
            actor_layer_sizes, critic_layer_sizes
        )
        self.episode_rewards = []

    def make_models(self, actor_layer_sizes, critic_layer_sizes):
        critic = (
            (
                nn.Sequential(
                    nn.Linear(
                        self.env.observation_space.shape[0],
                        critic_layer_sizes[0],
                    ),
                    nn.ReLU(),
                    nn.Linear(critic_layer_sizes[0], critic_layer_sizes[1]),
                    nn.ReLU(),
                    nn.Linear(critic_layer_sizes[1], 1),
                )
            )
            .to(device)
            .to(dtype)
        )

        actor = (
            (
                nn.Sequential(
                    nn.Linear(
                        self.env.observation_space.shape[0],
                        actor_layer_sizes[0],
                    ),
                    nn.Tanh(),
                    nn.Linear(actor_layer_sizes[0], actor_layer_sizes[1]),
                    nn.Tanh(),
                    nn.Linear(actor_layer_sizes[1], self.env.action_space.n),
                    nn.Softmax(dim=0),
                )
            )
            .to(device)
            .to(dtype)
        )

        return actor, critic

    def train(
        self, actor_lr, critic_lr, episodes, max_steps_per_episode, GAMMA=0.99
    ):
        actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        for e in range(episodes):
            state = self.env.reset()
            episode_reward = 0

            for _ in range(max_steps_per_episode):
                self.env.render()

                probs = self.actor(torch.from_numpy(state))
                action_distribution = Categorical(probs)
                action = action_distribution.sample()

                next_state, reward, done, _ = self.env.step(action.item())
                episode_reward += reward

                advantage = (
                    reward
                    + (1 - done)
                    * GAMMA
                    * self.critic(torch.from_numpy(next_state))
                    - self.critic(torch.from_numpy(state))
                )

                critic_loss = advantage.pow(2).mean()
                critic_loss.backward()
                critic_optimizer.step()
                critic_optimizer.zero_grad()

                actor_loss = (
                    -action_distribution.log_prob(action) * advantage.detach()
                )
                actor_loss.backward()
                actor_optimizer.step()
                actor_optimizer.zero_grad()

                state = next_state

                if done:
                    print(
                        "Completed episode {}/{} of training".format(
                            e, episodes
                        )
                    )
                    self.episode_rewards.append(episode_reward)
                    break

        self.env.close()

    def test(self, EPISODES, MAX_TIME_STEPS):
        for e in range(EPISODES):
            print("Episode: {}/{}".format(e+1, EPISODES))
            state = self.env.reset()

            for t in range(MAX_TIME_STEPS):
                print("\tTime step: {}/{}".format(t+1, MAX_TIME_STEPS))
                self.env.render()

                probs = self.actor(torch.from_numpy(state))
                action_distribution = Categorical(probs)
                action = action_distribution.sample()

                next_state, _, done, _ = self.env.step(action)

                state = next_state
                if done:
                    print("Episode {}/{} done".format(e, EPISODES))

        self.env.close()

    def plot(self):
        plt.plot(self.episode_rewards)
        plt.show()
