"""
Author : Vedant Shah
Email : vedantshah2012@gmail.com

Note : This can be used only for the KukaGymEnv environment
"""

import copy
from collections import deque, namedtuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import pybullet as p
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.max_buffer_size = max_buffer_size
        self.actor_layer_sizes = actor_layer_sizes
        self.critic_layer_sizes = critic_layer_sizes
        (
            self.Q,
            self.policy,
            self.Q_target,
            self.policy_target,
        ) = self.make_models(actor_layer_sizes, critic_layer_sizes)
        self.replay_buffer = deque(maxlen=max_buffer_size)

    def make_models(self, actor_layer_sizes, critic_layer_sizes):
        Q = (
                nn.Sequential(
                    nn.Linear(
                        self.env.observation_space.shape[0]
                        + 12
                        + self.env.action_space.shape[0],
                        critic_layer_sizes[0],
                    ),
                    nn.ReLU(),
                    nn.Linear(critic_layer_sizes[0], critic_layer_sizes[1]),
                    nn.ReLU(),
                    nn.Linear(critic_layer_sizes[1], 1),
                ).to(device).to(dtype)
            )

        policy = (
                nn.Sequential(
                    nn.Linear(
                        self.env.observation_space.shape[0] + 12,
                        actor_layer_sizes[0],
                    ),
                    nn.ReLU(),
                    nn.Linear(actor_layer_sizes[0], actor_layer_sizes[1]),
                    nn.ReLU(),
                    nn.Linear(
                        actor_layer_sizes[1], self.env.action_space.shape[0]
                    ),
                    nn.Tanh(),
                ).to(device).to(dtype)
            )
        

        Q_target = copy.deepcopy(Q)
        policy_target = copy.deepcopy(policy)

        return Q, policy, Q_target, policy_target

    def sample_batch(self, batch_size=64):
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

    def select_action(self, state, mean, std):
        with torch.no_grad():
          if (len(self.replay_buffer) <= 200):
            return torch.tensor(self.env.action_space.sample(), device=device, dtype=dtype).unsqueeze(0)
          empty_ten = torch.empty(self.env.action_space.shape).to(device).to(dtype)
          noisy_action = self.policy(state) + empty_ten.normal_(mean=mean, std=std)
          
          action = torch.clamp(
              noisy_action,
              self.env.action_space.low[0],
              self.env.action_space.high[0],
          ).to(device).to(dtype)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.replay_buffer) < self.max_buffer_size:
            self.replay_buffer.append(
                Transition(state, action, reward, next_state, done)
            )
        else:
            self.replay_buffer.popleft()
            self.replay_buffer.append(
                Transition(state, action, reward, next_state, done)
            )

    def update(
        self,
        polyak_constant,
        actor_optimizer,
        critic_optimizer,
        gamma,
        batch_size,
    ):
        sample_batch = self.sample_batch(batch_size)
        target = sample_batch.reward + (
            ~sample_batch.done
        ) * gamma * self.Q_target(
            torch.cat(
                (
                    sample_batch.next_state,
                    self.policy_target(sample_batch.next_state),
                ),
                dim=1,
            )
        )

        Q_loss = nn.MSELoss()(
            target,
            self.Q(
                torch.cat((sample_batch.state, sample_batch.action), dim=1)
            ),
        )
        critic_optimizer.zero_grad()
        Q_loss.backward()
        critic_optimizer.step()

        policy_loss = -1 * torch.mean(
            self.Q(
                torch.cat(
                    (sample_batch.state, self.policy(sample_batch.state)),
                    dim=1,
                )
            )
        )
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()

        for policy_param, policy_target_param in zip(
            self.policy.parameters(), self.policy_target.parameters()
        ):
            policy_target_param.data = (
                polyak_constant * policy_param.data
                + (1 - polyak_constant) * policy_target_param.data
            )

        for Q_param, Q_target_param in zip(
            self.Q.parameters(), self.Q_target.parameters()
        ):
            Q_target_param.data = (
                polyak_constant * Q_param.data
                + (1 - polyak_constant) * Q_target_param.data
            )
          
        return Q_loss, policy_loss

    def train(
        self,
        actor_lr=0.000005,
        critic_lr=0.00005,
        gamma=0.99,
        batch_size=32,
        mean=0,
        std=0.99,
        update_after=1,
        max_episodes=50,
        max_time_steps=1000,
        polyak_constant=0.001,
        RENDER=False,
        path_to_dir=None
    ):
        actor_optimizer = optim.Adam(self.policy.parameters(), lr=actor_lr)
        critic_optimizer = optim.Adam(self.Q.parameters(), lr=critic_lr)
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        training_rewards_list = []
        print("\nStarting Training:")

        for e in range(max_episodes):
            state = self.env.reset()
            state = torch.tensor(state, device=device, dtype=dtype).unsqueeze(
                0
            )
            blockPos, blockOrn = p.getBasePositionAndOrientation(
                self.env.blockUid
            )
            finalposangles = p.calculateInverseKinematics(
                self.env._kuka.kukaUid, 6, blockPos
            )
            finalposangles = torch.tensor(
                finalposangles, device=device, dtype=dtype
            ).unsqueeze(0)
            state = torch.cat((state, finalposangles), dim=1)
            episode_reward = 0
            experience_replay = []
            done = torch.tensor([0], dtype=torch.bool, device=device)
            t = -1
            print(f"\nStarting Episode : {e + 1}/{max_episodes}")

            for i in range(max_time_steps):
                t += 1

                if RENDER:
                    self.env.render()

                action = self.select_action(state, mean, std)
                next_state, reward, done, _ = self.env.step(action[0])
                episode_reward += reward

                next_state = torch.tensor(
                    next_state, device=device, dtype=dtype
                ).unsqueeze(0)
                reward = torch.tensor(
                    [reward], device=device, dtype=dtype
                ).unsqueeze(0)
                done = torch.tensor(
                    [done], device=device, dtype=torch.bool
                ).unsqueeze(0)

                actual_state = state[0][0:9].unsqueeze(0)
                experience_replay.append([actual_state, action, reward, next_state, done])
                next_state = torch.cat((next_state, finalposangles), dim=1)
                self.store_transition(state, action, reward, next_state, done)

                state = next_state

                print(
                    f"\tTime Step : {t + 1} of episode : {e + 1}/{max_episodes}"
                )

                if ((t + 1) % update_after == 0 and len(self.replay_buffer) > 200):
                    q_loss, policy_loss = self.update(
                        polyak_constant=polyak_constant,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        gamma=gamma,
                        batch_size=batch_size,
                    )
                    print(f"\t\tQ_loss = {q_loss}, policy_loss = {policy_loss}")

                if done[0].item() or (t + 1) == max_time_steps:
                    gripperState  = p.getLinkState(self.env._kuka.kukaUid, self.env._kuka.kukaGripperIndex)
                    gripperPos = gripperState[0]
                    final_state = p.calculateInverseKinematics(self.env._kuka.kukaUid, 6, gripperPos)
                    final_state = torch.tensor(final_state, device=device, dtype=dtype).unsqueeze(0)
                    for i, transition in enumerate(experience_replay):
                      if i == (len(experience_replay) - 1):
                        transition[2] = torch.tensor([1], device=device, dtype=dtype).unsqueeze(0)
                      transition[0] = torch.cat((transition[0], final_state), dim=1)
                      transition[3] = torch.cat((transition[3], final_state), dim=1)
                      self.store_transition(transition[0], transition[1], transition[2], transition[3], transition[4])
                    
                    training_rewards_list.append(episode_reward)
                    print(f"Episode {e + 1} completed; Episode Reward = {episode_reward}")
                    
                    break

        self.env.close()
        if path_to_dir is not None:
            self.save(path_to_dir)
        self.plot(rewards_list=training_rewards_list, file_name = "kuka_her_train.png")

    def save(self, path_to_dir):
        path1 = os.path.join(path_to_dir, "ddpg_her_policy.pth")
        path2 = os.path.join(path_to_dir, "ddpg_her_Q.pth")
        torch.save(self.policy.state_dict(), path1)
        torch.save(self.Q.state_dict(), path2)
    
    def evaluate(self, path_to_models, EPISODES=10):
        path1 = os.path.join(path_to_models, "ddpg_her_policy.pt")
        path2 = os.path.join(path_to_models, "ddpg_her_Q.pt")
        _, policy, _, _ = self.make_models(self.actor_layer_sizes, self.critic_layer_sizes)
        policy.load_state_dict(torch.load(path1))
        test_reward_list = []

        for e in range(EPISODES):
            print(f"STARTING EPISODE {e + 1} / {EPISODES}\n")
            done = False
            episode_reward = 0
            state = self.env.reset()
            state = torch.tensor(state, device=device, dtype=dtype)
            blockPos, blockOrn = p.getBasePositionAndOrientation(
                self.env.blockUid
            )
            finalposangles = p.calculateInverseKinematics(
                self.env._kuka.kukaUid, 6, blockPos
            )
            finalposangles = torch.tensor(
                finalposangles, device=device, dtype=dtype
            )
            state = torch.cat((state, finalposangles))
            t = 0

            while not done:
                t += 1
                print(f"\tTime step {t + 1}, Episode {e + 1} / {EPISODES}")
                self.env.render()
                action = policy(state)
                next_state, reward, done, _ = self.env.step(action.detach())
                episode_reward += reward

                next_state = torch.tensor(next_state, device=device, dtype=dtype)

                gripperState  = p.getLinkState(self.env._kuka.kukaUid, self.env._kuka.kukaGripperIndex)
                gripperPos = gripperState[0]
                state_extension = p.calculateInverseKinematics(self.env._kuka.kukaUid, 6, gripperPos)
                state_extension = torch.tensor(state_extension, device=device, dtype=dtype)

                next_state = torch.cat((next_state, state_extension))

                state = next_state

                if done:
                    print("\n")
                    break

            test_reward_list.append(episode_reward)

        self.env.close()
        self.plot(rewards_list=test_reward_list, filename="kuka_her_test.png")


    def plot(self, rewards_list, file_name):
        plt.plot(rewards_list)
        plt.savefig(file_name)
        plt.show()


if __name__ == "__main__":
    import pybullet_envs.bullet as bullet

    env = bullet.kukaGymEnv.KukaGymEnv(renders=False, isDiscrete=False)

    myagent = agent(env)
    myagent.train(max_episodes=20, max_time_steps=400, update_after=2, RENDER=False, path_to_dir="models")
    myagent.evaluate(EPISODES=5, path_to_models="models")

