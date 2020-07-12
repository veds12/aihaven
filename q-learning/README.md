# Q-Learning

Environment used - [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/)

## q_learning.py

```
from q-learning import q_learning.agent
my_agent = agent(STEP-SIZE, GAMMA, EPSILON, env)
my_agent.train
```

* STEP_SIZE - The parameter alpha, for updating the q-values
* GAMMA - Discount factor
* EPSILON - Parameter for epsilon-greedy action selection

## train.py

```
python3 train.py
```

* For training the q-values for each action state pair
* To change the parameter values open the script and change the values of STEP_SIZE (alpha), GAMMA (Discount Factor) and EPSILON.


## Current Tasks

- [ ] Add tqdm progress bar for training
- [ ] Test the learned q-values on an episode of the game

## References

* https://gym.openai.com/envs/FrozenLake-v0/
* https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#deadly-triad-issue

