'''class agent:
	def __init__(self, step_size, gamma, epsilon, env):
		self.q_table = np.random.rand(16,4)
		self.step_size = step_size
		self.gamma = gamma
		self.epsilon = epsilon
		self.env = env
		self.init_state = 0
		env.render()

	def epsilon_greedy(self):
		if (np.random.random() < self.epsilon):
			a = np.random.choice(range(4))
		else:
			a = np.argmax(self.q_table[self.init_state : ])
		return a

	def time_step(self):
		a = self.epsilon_greedy()
		next_state, reward, done, _ = env.step(a)
		self.update(a, next_state, reward)
		self.init_state = next_state
		return done

	def update(self, a, next_state, reward):
		self.q_table[self.init_state, a] = self.q_table[self.init_state, a] + self.step_size * (reward + self.gamma * np.amax(self.q_table[int(next_state) : ]) - self.q_table[self.init_state, a])

	def train(self):
		print(self.q_table)
		for _ in range(100):
			self.init_state = self.env.reset()
			for _ in range(1000):
				status = self.time_step()
				if status:
					print("Episode finished after certain timesteps")
					break
		print(self.q_table)
		env.close()'''

import gym
import numpy as np

dict = {
	0 : "LEFT",
	1 : "DOWN",
	2 : "RIGHT",
	3 : "UP"
}

class agent:
	def __init__(self, step_size, gamma, epsilon, env):
		self.q_table = np.random.rand(16,4)
		self.q_table[15] = [0, 0, 0, 0]
		self.step_size = step_size
		self.gamma = gamma
		self.epsilon = epsilon
		self.env = env
		self.init_state = 0
		env.render()

	def epsilon_greedy(self):
		#print("Selecting action")
		rass = np.random.random()
		if (rass < self.epsilon):
			#print("random number generated is 1:", rass)
			a = np.random.choice(range(4))
		else:
			#print("random number generated is 2:", rass)
			a = np.argmax(self.q_table[self.init_state])
		#print("Action selected:", dict[a])
		return a

	def time_step(self):
		#print("done1")
		a = self.epsilon_greedy()
		next_state, reward, done, _ = self.env.step(a)
		#print("Next state and reward respectively are :", next_state, reward)
		self.update(a, next_state, reward)
		self.init_state = next_state
		return done

	def update(self, a, next_state, reward):
		#print("Updating the q-value of state:", self.init_state)
		self.q_table[self.init_state, a] = self.q_table[self.init_state, a] + self.step_size * (reward + self.gamma * np.max(self.q_table[int(next_state)]) - self.q_table[self.init_state, a])
		#print("New value of the initial state and action :", a, self.q_table[self.init_state, a])

	def train(self):
		print("Initial q-table:\n")
		print(self.q_table,"\n")
		#print("Entered train function. Starting training over 100 episodes")
		for i in range(1000	):
			#print("Episode :", i)
			self.init_state = self.env.reset()
			#print("Initial state is :", self.init_state)
			for j in range(1000):
				status = self.time_step()
				if status:
					break
		print("Final q-table:\n")
		print(self.q_table)
		self.env.close()



		




