import numpy as np

class QLearningAgent:
    def __init__(self, action_space, state_space, grid_size):
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.action_space = action_space
        self.grid_size = grid_size  # Add grid_size here
        self.q_table = np.zeros((self.grid_size, self.grid_size, action_space.n))  # Now this works


    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # Exploration
        else:
            # Exploitation: Choose action with highest Q-value
            return np.argmax(self.q_table[state[0], state[1], :])
    def update_q_table(self, state, action, reward, next_state):
    # Extract the coordinates of the current state and next state
        x, y = state
        next_x, next_y = next_state
        
        # Get the best action for the next state (this is the action with the maximum Q-value)
        best_next_action = np.argmax(self.q_table[next_x, next_y, :])
        
        # Update Q-value using the Q-learning formula
        self.q_table[x, y, action] = (1 - self.alpha) * self.q_table[x, y, action] + \
                                    self.alpha * (reward + self.gamma * self.q_table[next_x, next_y, best_next_action])

