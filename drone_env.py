import gym
import numpy as np
from gym import spaces

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        
        self.grid_size = 10  # 10x10 grid
        self.state = np.array([0, 0])  # Start at (0,0)
        self.goal = np.array([9, 9])  # Goal at (9,9)
        self.obstacles = [np.array([5, 5]), np.array([6, 6])]  # Example obstacles
        
        self.action_space = spaces.Discrete(4)  # 4 possible moves: Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
    
    def reset(self):
        self.state = np.array([0, 0])
        return self.state
    
    def step(self, action):
        new_state = self.state.copy()
        
        if action == 0:  # Move Up
            new_state[1] = min(self.grid_size - 1, self.state[1] + 1)
        elif action == 1:  # Move Down
            new_state[1] = max(0, self.state[1] - 1)
        elif action == 2:  # Move Left
            new_state[0] = max(0, self.state[0] - 1)
        elif action == 3:  # Move Right
            new_state[0] = min(self.grid_size - 1, self.state[0] + 1)
        
        reward = -1  # Small penalty for each move
        done = False
        
        if any(np.array_equal(new_state, obs) for obs in self.obstacles):
            reward = -100  # Hit an obstacle
            done = True
        elif np.array_equal(new_state, self.goal):
            reward = 100  # Reached goal
            done = True
        
        self.state = new_state
        return self.state, reward, done, {}
    
    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:, :] = '.'
        grid[self.state[1], self.state[0]] = 'D'  # Drone
        grid[self.goal[1], self.goal[0]] = 'G'  # Goal
        for obs in self.obstacles:
            grid[obs[1], obs[0]] = 'X'  # Obstacles
        
        print("\n".join(" ".join(row) for row in grid))
        print("\n")

# Quick test
if __name__ == "__main__":
    env = DroneEnv()
    obs = env.reset()
    done = False
    while not done:
        action = np.random.choice(4)  # Random action for testing
        obs, reward, done, _ = env.step(action)
        env.render()
