import numpy as np

class QLearningAgent:
    def __init__(self, action_space, grid_size):
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.action_space = action_space
        self.grid_size = grid_size  # Ensure grid size is dynamic
        self.q_table = np.zeros((self.grid_size, self.grid_size, action_space.n))

    def select_action(self, state):
        x, y = state  # Ensure state is handled as a tuple
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # Exploration
        else:
            return np.argmax(self.q_table[x, y, :])  # Exploitation

    def update_q_table(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        best_next_action = np.argmax(self.q_table[next_x, next_y, :])
        self.q_table[x, y, action] = (1 - self.alpha) * self.q_table[x, y, action] + \
                                     self.alpha * (reward + self.gamma * self.q_table[next_x, next_y, best_next_action])
