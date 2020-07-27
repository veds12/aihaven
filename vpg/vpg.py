import gym
import torch
import torch.distributions.categorical as Categorical
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class agent:
    def __init__(self, env, policy_layers_sizes, value_layers_sizes):

        self.env = env
        self.policy_network, self.value_network = self.make_models(
            policy_layers_sizes, value_layers_sizes
        )

    def make_models(
        self, policy_layers_sizes=[32, 32], value_layers_sizes=[24, 24]
    ):

        policy_network = (
            (
                nn.Sequential(
                    nn.Linear(
                        self.env.observation.space.shape[0],
                        policy_layers_sizes[0],
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        policy_layers_sizes[0], policy_layers_sizes[1]
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        policy_layers_sizes[1], self.env.action_space.n
                    ),
                    nn.Softmax(dim=0),
                )
            )
            .to(device)
            .to(dtype)
        )

        value_network = (
            (
                nn.Sequential(
                    nn.Linear(
                        self.env.observation_space.shape[0],
                        value_layers_sizes[0],
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        value_layers_sizes[0], value_layers_sizes[1]
                    ),
                    nn.ReLU(),
                    nn.Linear(value_layers_sizes[1], 1),
                )
            )
            .to(device)
            .to(dtype)
        )
        return policy_network, value_network

    def train(self, epochs, episodes, max_steps, v_lr, p_lr, gamma=0.9):

        for E in range(epochs):
            p_loss = torch.tensor([[0]], device=device, dtype=dtype)
            v_loss = torch.tensor([[0]], device=device, dtype=dtype)
            p_optimizer = optim.Adam(
                self.policy_network.parameters(), lr=p_lr
            )
            v_optimizer = optim.Adam(
                self.value_network.parameters(), lr=v_lr
            )

            for e in range(episodes):
                observation = self.env.reset()
                observation = torch.tensor(
                    observation, device=device, dtype=dtype
                )
                obsrv_traj = [observation]
                reward_traj = []
                value_traj = [self.value_network(observation)]
                rtg = [0]
                policy_distribution_traj = []
                action_traj = []

                for _ in range(max_steps):
                    self.env.render()
                    policy_distribution = Categorical(
                        self.policy_network(observation)
                    )
                    action = policy_distribution.sample()
                    observation, reward, done, _ = self.env.step(
                        action.item()
                    )
                    observation = torch.tensor(
                        observation, device=device, dtype=dtype
                    )
                    reward = torch.tensor(
                        reward, device=device, dtype=dtype
                    )
                    obsrv_traj.append(observation)
                    reward_traj.append(reward)
                    policy_distribution_traj.append(policy_distribution)
                    action_traj.append(action)

                    if done:
                        reversed(reward_traj)
                        for j, r in enumerate(reward_traj):
                            rtg.append(r + gamma * rtg[j])
                        reversed(rtg)
                        reversed(reward_traj)
                        break
                    else:
                        value_traj.append(self.value_network(observation))

                obsrv_traj = torch.tensor(
                    obsrv_traj, device=device, dtype=dtype
                )
                value_traj = torch.tensor(
                    value_traj, device=device, dtype=dtype
                )
                policy_distribution_traj = torch.tensor(
                    policy_distribution_traj, device=device, dtype=dtype
                )
                action_traj = torch.tensor(
                    action_traj, device=device, dtype=dtype
                )
                rtg = torch.tensor(rtg, device=device, dtype=dtype)

                assert value_traj.size == rtg.size

                advantage = rtg - value_traj

                v_loss += nn.MSELoss()(rtg, value_traj)
                v_loss = v_loss / len(reward_traj)

                for k in range(len(reward_traj) - 1):
                    p_loss -= (
                        policy_distribution_traj[k].log_prob(
                            action_traj[k]
                        )
                        * advantage[k]
                    )

                print(
                    "Episode {}/{}, Epoch {}/{}".format(
                        e, episodes, E, epochs
                    )
                )

            p_loss = p_loss / episodes
            v_loss = v_loss / episodes

            p_loss.backward()
            p_optimizer.step()
            v_loss.backward()
            v_optimizer.step()

        self.env.close()

    def test(self, EPISODES, MAX_STEPS):

        for e in range(EPISODES):
            observation = self.env.reset()

            for _ in range(MAX_STEPS):
                self.env.render()
                policy_distribution = Categorical(
                    self.policy_network(observation)
                )
                action = policy_distribution.sample()
                observation, _, done, _ = self.env.step(action.item())
                if done:
                    print("Episode {}/{}finished".format(e, EPISODES))
        self.env.close()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    POLICY_LAYER_SIZES = [32, 32]
    VALUE_LAYER_SIZES = [32, 32]
    EPOCHS = 500
    EPISODES = 5
    MAX_STEPS = 1000
    V_lr = 0.01
    P_lr = 0.001
    GAMMA = 0.99

    myagent = agent(env, POLICY_LAYER_SIZES, VALUE_LAYER_SIZES)
    myagent.train(EPOCHS, EPISODES, MAX_STEPS, V_lr, P_lr, GAMMA)

    TEST_EPISODES = 4
    TEST_MAX_STEPS = 1000

    myagent.test(TEST_EPISODES, TEST_MAX_STEPS)
