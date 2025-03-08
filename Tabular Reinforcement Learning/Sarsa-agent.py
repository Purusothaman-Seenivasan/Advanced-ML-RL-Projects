import numpy as np

class Agent(object):
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.q_table = np.random.uniform(size=(state_space, action_space))
        self.q_table[-1] = np.zeros(action_space)
        self.epsilon = 0.05
        self.gamma = 0.95
        self.action = 0
        self.alpha = 0.05
        self.state = 0
    
    def observe(self, observation, reward, done):
        new_action = self.act(observation)
        if done:
            new_q_value_difference = self.alpha * (reward  - self.q_table[self.state, self.action])
        else:
            new_q_value_difference = self.alpha * (reward + self.gamma * self.q_table[observation, new_action] - self.q_table[self.state, self.action])
        self.q_table[self.state, self.action] +=  new_q_value_difference
        self.action = new_action

    def act(self, observation):
        greedy = np.random.uniform()
        if greedy > self.epsilon:
            max_val = np.max(self.q_table[observation])
            max_val_indices = np.where(self.q_table[observation] == max_val)[0]
            action = np.random.choice(max_val_indices)
        else:
            action = np.random.randint(self.action_space)
        return action