import numpy as np

class Agent(object):
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.q_table_1 = np.random.uniform(size=(state_space, action_space))
        self.q_table_1[-1] = np.zeros(action_space)
        self.q_table_2 = np.random.uniform(size=(state_space, action_space))
        self.q_table_2[-1] = np.zeros(action_space)
        self.epsilon = 0.05
        self.gamma = 0.95
        self.action = 0
        self.alpha = 0.05
        self.state = 0
    
    def observe(self, observation, reward, done):
        choose_table = np.random.uniform()
        if choose_table <= 0.5:
            max_val = np.max(self.q_table_1[observation])
            max_val_indices = np.where(self.q_table_1[observation] == max_val)[0]
            new_action = np.random.choice(max_val_indices)
            if done:
                new_q_value_difference = self.alpha * (reward  - self.q_table_1[self.state, self.action])
            else:
                new_q_value_difference = self.alpha * (reward + self.gamma * self.q_table_2[observation, new_action] - self.q_table_1[self.state, self.action])
            self.q_table_1[self.state, self.action] +=  new_q_value_difference
        else:
            max_val = np.max(self.q_table_2[observation])
            max_val_indices = np.where(self.q_table_2[observation] == max_val)[0]
            new_action = np.random.choice(max_val_indices)
            if done:
                new_q_value_difference = self.alpha * (reward  - self.q_table_2[self.state, self.action])
            else:
                new_q_value_difference = self.alpha * (reward + self.gamma * self.q_table_1[observation, new_action] - self.q_table_2[self.state, self.action])
            self.q_table_2[self.state, self.action] +=  new_q_value_difference

    def act(self, observation):
        greedy = np.random.uniform()
        if greedy > self.epsilon:
            q1_plus_q2 = self.q_table_1[observation] + self.q_table_2[observation]
            max_val = np.max(q1_plus_q2)
            max_val_indices = np.where(q1_plus_q2 == max_val)[0]
            action = np.random.choice(max_val_indices)
        else:
            action = np.random.randint(self.action_space)
        return action