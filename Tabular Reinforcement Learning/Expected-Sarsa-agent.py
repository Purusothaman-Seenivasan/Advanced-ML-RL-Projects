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
        expected_value = 0
        max_val = np.max(self.q_table[observation])
        max_val_indices = np.where(self.q_table[observation] == max_val)[0]
        prob_arg_max = (1 - self.epsilon)  * (1 / len(max_val_indices))
        prob_random_action = self.epsilon * (1 / self.action_space)
        for i in range(self.action_space):
            expected_value += prob_random_action * self.q_table[observation, i]
            if i in max_val_indices:
                expected_value += prob_arg_max * self.q_table[observation, i]
        if done:
            new_q_value_difference = self.alpha * (reward  - self.q_table[self.state, self.action])
        else:
            new_q_value_difference = self.alpha * (reward + self.gamma * expected_value - self.q_table[self.state, self.action])
        self.q_table[self.state, self.action] +=  new_q_value_difference

    def act(self, observation):
        greedy = np.random.uniform()
        if greedy > self.epsilon:
            max_val = np.max(self.q_table[observation])
            max_val_indices = np.where(self.q_table[observation] == max_val)[0]
            action = np.random.choice(max_val_indices)
        else:
            action = np.random.randint(self.action_space)
        return action