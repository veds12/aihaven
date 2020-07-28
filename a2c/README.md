# Advantage Actor Critic (A2C)

* Framework : PyTorch
* The actor and critic function approximators are two hidden layer neural networks

## Usage

```
from aihaven.a2c import agent
my_agent = agent(env, actor_layer_sizes, critic_layer_sizes)
my_agent.train(actor_lr, critic_lr, episodes, max_steps_per_episode, GAMMA)
my_agent.plot()
my_agent.train(EPISODES)
````

### Attributes

* **agent()** - Agent class
    * **env** - Gym environment
    * **actor_layer_sizes** - Two element list specifying the sizes of the first and second hidden layers of the actor (default = [64, 32])
    * **critic_layer_sizes** - Two element list specifying the sizes of the first and second hidden layers of the critic (default = [64, 32])
  
* **agent.train()**
    * **actor_lr** - Learning rate for actor network optimizer
    * **critic_lr** - Learning rate for critic network optimizer
    * **episodes** - Number of episodes to train the agent on
    * **max_steps_per_episode** - Maximum time steps per episode to train the agent on
    * **GAMMA** - Discount factor
  
* **agent.plot** - Plot the rewards as the agent trains
  
* **agent.test()** - Test the train agents
    * **EPISODES** - Number of episodes to run the test for

## To Do

- [ ] Tune hyperparameters
- [ ] Add plots 

## References

* https://www.youtube.com/watch?v=bRfUxQs6xIM&t=4347s

