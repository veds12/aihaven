"""
Author : Vedant Shah
Email : vedantshah2012@gmail.com

"""

import copy
from collections import deque, namedtuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import pybullet as p

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
        critic_loss.backward()
        critic_optimizer.step()

        policy_loss = -1 * torch.mean(
            self.Q(
                torch.cat(
                    (sample_batch.state, self.policy(sample_batch.state)),
                    dim=1,
                )
            )
        )
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

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
    ):
        actor_optimizer = optim.Adam(self.policy.parameters(), lr=actor_lr)
        critic_optimizer = optim.Adam(self.Q.parameters(), lr=critic_lr)
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        self.training_rewards_list = []
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

            while not done[0].item():
                t += 1
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

                if (t + 1) % update_after == 1:
                    self.update(
                        polyak_constant=polyak_constant,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        gamma=gamma,
                        batch_size=batch_size,
                    )

                print(
                    f"\tTime Step : {t + 1} of episode : {e + 1}/{max_episodes}"
                )

                if done[0].item():
                    gripperState  = p.getLinkState(self.env._kuka.kukaUid, self.env._kuka.kukaGripperIndex)
                    gripperPos = gripperState[0]
                    final_state = p.calculateInverseKinematics(self.env._kuka.kukaUid, 6, gripperPos)
                    final_state = torch.tensor(final_state, device=device, dtype=dtype).unsqueeze(0)
                    for i, transition in enumerate(experience_replay):
                      if i == (len(experience_replay) - 1):
                        transition[2] = torch.tensor([1], device=device, dtype=dtype)
                      transition[0] = torch.cat((transition[0], final_state), dim=1)
                      transition[3] = torch.cat((transition[3], final_state), dim=1)
                      self.store_transition(transition[0], transition[1], transition[2], transition[3], transition[4])
                    
                    break

            self.training_rewards_list.append(episode_reward)

        self.env.close()
        self.plot()

    def plot(self):
        plt.plot(self.training_rewards_list)
        plt.savefig("./plots/kuka_her_training.png")
        plt.show()


if __name__ == "__main__":
    import pybullet_envs.bullet as bullet

    env = bullet.kukaGymEnv.KukaGymEnv(renders=False, isDiscrete=False)

    myagent = agent(env)
    myagent.train(max_episodes=50, max_time_steps=700)
