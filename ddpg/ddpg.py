# Author : Vedant Shah
# E-mail : vedantshah2012@gmail.com

import copy
import random
from collections import deque, namedtuple

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

"""
Hyperparameters:

actor_layer_sizes
critic_layer_sizes
max_buffer_size
polyak_constant
max_time_steps
max_episodes
actor_lr
critic_lr
GAMMA
update_after
batch_size
mean
std
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class agent:
    def __init__(
        self,
        env,
        actor_layer_sizes=[32, 32],
        critic_layer_sizes=[64, 64],
        max_buffer_size=int(1e6),
    ):
        self.env = env
        (
            self.actor,
            self.critic,
            self.target_actor,
            self.target_critic,
        ) = self.make_models(actor_layer_sizes, critic_layer_sizes)
        self.replay_buffer = deque(maxlen=max_buffer_size)
        self.max_buffer_size = max_buffer_size

    def make_models(self, actor_layer_sizes, critic_layer_sizes):
        actor = (
            nn.Sequential(
                nn.Linear(
                    self.env.observation_space.shape[0], actor_layer_sizes[0],
                ),
                nn.ReLU(),
                nn.Linear(actor_layer_sizes[0], actor_layer_sizes[1]),
                nn.ReLU(),
                nn.Linear(
                    actor_layer_sizes[1], self.env.action_space.shape[0]
                ),
                nn.Tanh(),
            )
            .to(device)
            .to(dtype)
        )

        critic = (
            nn.Sequential(
                nn.Linear(
                    self.env.observation_space.shape[0]
                    + self.env.action_space.shape[0],
                    critic_layer_sizes[0],
                ),
                nn.ReLU(),
                nn.Linear(critic_layer_sizes[0], critic_layer_sizes[1]),
                nn.Linear(critic_layer_sizes[1], 1),
            )
            .to(device)
            .to(dtype)
        )

        target_actor = copy.deepcopy(actor)  # Create a target actor network

        target_critic = copy.deepcopy(
            critic
        )  # Create a target critic network

        return actor, critic, target_actor, target_critic

    def select_action(
        self, state, mean=0, std=0.99
    ):  # Selects an action in exploratory manner
        with torch.no_grad():
            noisy_action = self.actor(state) + torch.empty(
                self.env.action_space.shape
            ).normal_(mean=mean, std=std)
            action = torch.clamp(
                noisy_action,
                self.env.action_space.low[0],
                self.env.action_space.high[0],
            )

        return action

    def store_transition(
        self, state, action, reward, next_state, done
    ):  # Stores the transition to the replay buffer with a default maximum capacity of 2500
        if len(self.replay_buffer) < self.max_buffer_size:
            self.replay_buffer.append(
                Transition(state, action, reward, next_state, done)
            )
        else:
            self.replay_buffer.popleft()
            self.replay_buffer.append(
                Transition(state, action, reward, next_state, done)
            )

    def sample_batch(
        self, batch_size=64
    ):  # Samples a random batch of transitions for training
        return Transition(
            *[
                torch.cat(i)
                for i in [
                    *zip(
                        *random.sample(
                            self.replay_buffer,
                            min(len(self.replay_buffer), batch_size),
                        )
                    )
                ]
            ]
        )

    def train(
        self,
        GAMMA=0.99,
        actor_lr=0.001,
        critic_lr=0.001,
        polyak_constant=0.001,
        max_time_steps=6000,
        max_episodes=300,
        update_after=1,
        batch_size=64,
        mean=0,
        std=0.99,
    ):
        self.train_rewards_list = []
        actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        print("Starting Training:\n")
        for e in range(max_episodes):

            state = self.env.reset()
            state = torch.tensor(state, device=device, dtype=dtype).unsqueeze(
                0
            )
            episode_reward = 0

            for t in range(max_time_steps):
                self.env.render()
                action = self.select_action(state, mean=mean, std=std)
                next_state, reward, done, _ = self.env.step(
                    action[0]
                )  # Sample a transition
                episode_reward += reward

                next_state = torch.tensor(
                    next_state, device=device, dtype=dtype
                ).unsqueeze(0)
                reward = torch.tensor(
                    [reward], device=device, dtype=dtype
                ).unsqueeze(0)
                done = torch.tensor(
                    [done], device=device, dtype=dtype
                ).unsqueeze(0)

                self.store_transition(
                    state, action, reward, next_state, done
                )  # Store the transition in the replay buffer

                state = next_state

                sample_batch = self.sample_batch(batch_size)

                with torch.no_grad():  # Determine the target for the critic to train on
                    target = sample_batch.reward + (
                        1 - sample_batch.done
                    ) * GAMMA * self.target_critic(
                        torch.cat(
                            (
                                sample_batch.next_state,
                                self.target_actor(sample_batch.next_state),
                            ),
                            dim=1,
                        )
                    )

                # Train the critic on the sampled batch
                critic_loss = nn.MSELoss()(
                    target,
                    self.critic(
                        torch.cat(
                            (sample_batch.state, sample_batch.action), dim=1
                        )
                    ),
                )

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss = -1 * torch.mean(
                    self.critic(
                        torch.cat(
                            (
                                sample_batch.state,
                                self.actor(sample_batch.state),
                            ),
                            dim=1,
                        )
                    )
                )

                # Train the actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # if (((t + 1) % update_after) == 0):
                for actor_param, target_actor_param in zip(
                    self.actor.parameters(), self.target_actor.parameters()
                ):
                    target_actor_param.data = (
                        polyak_constant * actor_param.data
                        + (1 - polyak_constant) * target_actor_param.data
                    )

                for critic_param, target_critic_param in zip(
                    self.critic.parameters(), self.target_critic.parameters()
                ):
                    target_critic_param.data = (
                        polyak_constant * critic_param.data
                        + (1 - polyak_constant) * target_critic_param.data
                    )

                print(
                    "\tTimestep {}/{} of episode {}".format(
                        t + 1, max_time_steps, e + 1
                    )
                )
                if done:
                    print(
                        "Completed episode {}/{}".format(e + 1, max_episodes)
                    )
                    break

            self.train_rewards_list.append(episode_reward)

        self.env.close()

    """def test(self, TEST_EPISODES=20):
        self.test_rewards_list = []
        print("\nTest begin:")
        for e in range(TEST_EPISODES):
            episode_reward = 0
            state = self.env.reset()
            state = torch.tensor(state, device=device, dtype=dtype).unsqueeze(0)
            done = 0
            t = 0
            while not done:
                t += 1
                self.env.render()
                action = self.actor(state)
                next_state, reward, done, _ = self.env.step(action[0])
                episode_reward += reward
                next_state = torch.tensor(next_state, device=device, dtype=dtype).unsqueeze(0)
                state = next_state
                print("Time step {} of episode {}/{}".format(t + 1, e + 1, EPISODES))
                if done:
                    print("Episode {} completed in {} timesteps".format(e + 1, t + 1))
                    break
            self.test_rewards_list.append(episode_reward)
        self.env.close()"""

    def plot(self, plot_type):
        if plot_type == "train":
            plt.plot(self.train_rewards_list)
            plt.show()
        elif plot_type == "test":
            plt.plot(self.test_rewards_list)
            plt.show()
        else:
            print("\nInvalid plot type")


if __name__ == "__main__":
    #import gym
    import pybullet_envs.bullet as bullet

    env = bullet.kukaGymEnv.KukaGymEnv(renders=False, isDiscrete=False)
    #env = gym.make("MountainCarContinuous-v0")
    #env._max_episode_steps = 7000
    myagent = agent(env)
    myagent.train()
    myagent.plot("train")

    # myagent.test()
    # myagent.plot("test")
