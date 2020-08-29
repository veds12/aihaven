import copy
import random
from collections import deque, namedtuple

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import pybullet as p

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

device = torch.device("cpu")
dtype = torch.double

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class agent:
    def __init__(
        self,
        env,
        actor_layer_sizes=[256, 128],
        critic_layer_sizes=[256, 128],
        max_buffer_size=50000,
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

        target_actor = copy.deepcopy(actor).to(device).to(dtype)  # Create a target actor network

        target_critic = copy.deepcopy(
            critic
        ).to(device).to(dtype)  # Create a target critic network

        return actor, critic, target_actor, target_critic

    def select_action(
        self, state, mean=0, std=0.99
    ):  # Selects an action in exploratory manner
        with torch.no_grad():
            empty_ten = torch.empty(self.env.action_space.shape).to(device).to(dtype)
            noisy_action = self.actor(state) + empty_ten.normal_(mean=mean, std=std)
          
            action = torch.clamp(
                noisy_action,
                self.env.action_space.low[0],
                self.env.action_space.high[0],
            ).to(device).to(dtype)

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
        actor_lr=0.000005,
        critic_lr=0.00005,
        polyak_constant=0.001,
        max_time_steps=4000,
        max_episodes=200,
        update_after=1,
        batch_size=32,
        mean=0,
        std=0.7,
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
            done = torch.tensor([0], device=device, dtype=dtype).unsqueeze(0)
            t = 0
            while not done[0].item():
                self.env.render()
                action = self.select_action(state, mean=mean, std=std)
                next_state, reward, done, _ = self.env.step(
                    action[0].cpu()
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

                # if (((t + 1) % update_after) == 0):

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

                    target = torch.tensor(target, device=device, dtype=dtype)

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
                    "\tTimestep {} of episode {}".format(
                        t + 1, e + 1
                    )
                )
                t += 1
                if done:
                    print(
                        "Completed episode {}/{}".format(e + 1, max_episodes)
                    )
                    break

            self.train_rewards_list.append(episode_reward)
        self.save()
        self.env.close()

    def save(self, path_to_dir=None):
        if path_to_dir is None:
            path_to_dir = "./models"
        path1 = os.path.join(path_to_dir, "actor.pth")
        path2 = os.path.join(path_to_dir, "critic.pth")
        torch.save(self.actor, path1)
        torch.save(self.critic, path2)

    def plot(self, plot_type):
        if (plot_type == "train"):
            plt.plot(self.train_rewards_list)
            plt.savefig("./plots/train_rewards_plot.png")
            plt.show()

        elif (plot_type == "test"):
            plt.plot(self.test_rewards_list)
            plt.savefig("./plots/test_rewards_plot.png")
            plt.show()

        else:
            print("Invalid plot type!")

    def test(self, episodes=5, path_to_models=None):
        if path_to_models is None:
            path_to_models = "./models"
        path1 = os.path.join(path_to_models, "actor.pth")
        path2 = os.path.join(path_to_models, "critic.pth")

        actor = torch.load(path1)
        self.test_rewards_list = []

        for e in range(episodes):
            state = self.env.reset()
            state = torch.tensor(state, device=device, dtype=dtype)
            done = 0
            episode_reward = 0
            while not done:
                self.env.render()
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action[0])
                episode_reward += reward

                next_state = torch.tensor(next_state, device=device, dtype=dtype)

                if done:
                    print(f"Completed episode {e + 1}/{episodes}")

            self.test_rewards_list.append(episode_reward)

        self.env.close()


if __name__ == "__main__":
    #import gym
    import pybullet_envs.bullet as bullet

    env = bullet.kukaGymEnv.KukaGymEnv(renders=False, isDiscrete=False)
    #env = gym.make("MountainCarContinuous-v0")
    #env._max_episode_steps = 7000
    myagent = agent(env)
    myagent.train(max_episodes=25)
    myagent.plot("train")

    # myagent.test()
    # myagent.plot("test")
