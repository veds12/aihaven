import gym
import numpy as np

dict = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}


class agent:
    def __init__(self, step_size, gamma, epsilon, env):
        self.q_table = np.random.rand(16, 4)
        self.q_table[15] = [0, 0, 0, 0]
        self.step_size = step_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = env
        self.init_state = 0
        env.render()

    def epsilon_greedy(self):
        # print("Selecting action")
        rass = np.random.random()
        if rass < self.epsilon:
            # print("random number generated is 1:", rass)
            a = np.random.choice(range(4))
        else:
            # print("random number generated is 2:", rass)
            a = np.argmax(self.q_table[self.init_state])
        # print("Action selected:", dict[a])
        return a

    def time_step(self):
        # print("done1")
        a = self.epsilon_greedy()
        next_state, reward, done, _ = self.env.step(a)
        # print("Next state and reward respectively are :", next_state, reward)
        self.update(a, next_state, reward, done)
        self.init_state = next_state
        return done

    def update(self, a, next_state, reward, done):
        # print("Updating the q-value of state:", self.init_state)
        self.q_table[self.init_state, a] = self.q_table[
            self.init_state, a
        ] + self.step_size * (
            reward
            + self.gamma * np.max(self.q_table[int(next_state)]) * (1 - done)
            - self.q_table[self.init_state, a]
        )
        # print("New value of the initial state and action :", a, self.q_table[self.init_state, a])

    def train(self):
        """ Train the agent using q-learning"""
        print("Initial q-table:\n")
        print(self.q_table, "\n")
        # print("Entered train function. Starting training over 100 episodes")
        for i in range(1000):
            # print("Episode :", i)
            self.init_state = self.env.reset()
            # print("Initial state is :", self.init_state)
            for j in range(1000):
                status = self.time_step()
                if status:
                    break
        print("Final q-table:\n")
        print(self.q_table)
        self.env.close()

    def display_value(self):
        """
		Displays the q- values for each state
		"""
        val = np.zeros((16, 1))
        for i in range(16):
            val[i] = np.max(self.q_table[i])
        val = np.reshape(val, (4, 4))
        print(val)
