# Deep Q-Network

* Framework used - PyTorch
* A two Deep Q-Network with two hidden layers
* Uses Memory Replay

## Usage 

```python
from aihaven.dqn.dqn import agent
import gym

env = gym.make('CartPole-v1')
myagent = agent(env, GAMMA, EPSILON, hidden_layers_size, buffer_size)
myagent.play(EPISODES, lr)
myagent.test(episodes)
```

## Attributes:

* **env** : An OpenAI gym environment
* **GAMMA** : Discount Factor
* **EPSILON** : Parameter for epsilon-greedy action selection
* **hidden_layers_size** : A two element list containing the sizes of the two hidden layers of the DQN
* **buffer_size** : Number of previous transitions that the agent remembers at a time
* **agent.play(EPISODES, lr)** : Train the agent for 'EPISODES' number of episodes and a learning rate 'lr' used for the optimizer
* **agent.test(episodes)** : Test the trained agent by running it for 'episode' number of episodes

## To Do:

- [ ] Add plots

## References:

* https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

