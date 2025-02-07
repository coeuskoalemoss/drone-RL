import gym
import numpy as np
from gym import spaces

class DroneEnv(gym.Env):
    def __init__(self, grid_size=10, num_obstacles=5):
        super(DroneEnv, self).__init__()
        
        self.grid_size = grid_size  # Dynamic grid size
        self.num_obstacles = num_obstacles  # Number of obstacles
        
        self.action_space = spaces.Discrete(4)  # 4 possible moves: Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)
        
        self.reset()
    
    def reset(self):
        self.state = np.array([0, 0])  # Always start at (0,0)
        self.goal = np.random.randint(0, self.grid_size, size=(2,))  # Random goal
        
        # Generate random obstacles that are not at the start or goal
        self.obstacles = []
        while len(self.obstacles) < self.num_obstacles:
            obs = tuple(np.random.randint(0, self.grid_size, size=(2,)))
            if obs != tuple(self.state) and obs != tuple(self.goal):
                self.obstacles.append(obs)
        
        return tuple(self.state)
    
    def step(self, action):
        new_state = self.state.copy()

        # Move the drone based on action
        if action == 0:  # Move Up
            new_state[1] = min(self.grid_size - 1, self.state[1] + 1)
        elif action == 1:  # Move Down
            new_state[1] = max(0, self.state[1] - 1)
        elif action == 2:  # Move Left
            new_state[0] = max(0, self.state[0] - 1)
        elif action == 3:  # Move Right
            new_state[0] = min(self.grid_size - 1, self.state[0] + 1)

        # Compute distance before and after move
        dist_old = np.linalg.norm(self.state - self.goal)  
        dist_new = np.linalg.norm(new_state - self.goal)

        reward = -1  # Small movement penalty
        done = False  # Always initialize done

        # Encourage movement toward the goal
        if dist_new < dist_old:
            reward += 5  # Reward for getting closer

        # Handle collisions with obstacles
        if tuple(new_state) in self.obstacles:
            reward = -50  # Reduced penalty for obstacle hit
            new_state = self.state  # Stay in place

        # Goal reached
        elif np.array_equal(new_state, self.goal):
            reward = 100  # Big reward
            done = True  # End episode

        self.state = new_state
        return tuple(self.state), reward, done, {}


        
    def render(self):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        grid[self.state[1], self.state[0]] = 'D'  # Drone
        grid[self.goal[1], self.goal[0]] = 'G'  # Goal
        for obs in self.obstacles:
            grid[obs[1], obs[0]] = 'X'  # Obstacles
        
        print("\n".join(" ".join(row) for row in grid))
        print("\n")

# Quick test
if __name__ == "__main__":
    env = DroneEnv(grid_size=10, num_obstacles=5)
    obs = env.reset()
    done = False
    while not done:
        action = np.random.choice(4)  # Random action for testing
        obs, reward, done, _ = env.step(action)
        env.render()
 # Add this inside the while loop in test.py
