import numpy as np
from drone_env import DroneEnv
from agent import QLearningAgent

# Initialize environment and agent
env = DroneEnv()
agent = QLearningAgent(action_space=env.action_space, grid_size=env.grid_size)

# Hyperparameters
episodes = 1000
max_steps = 100
epsilon = 0.1  # Initial exploration rate

# Training loop
for episode in range(episodes):
    state = tuple(env.reset())  # Ensure state is a tuple
    done = False
    total_reward = 0
    
    # Decay epsilon over time
    epsilon = max(0.05, epsilon * 0.997)  # Slower decay, minimum 0.05


    agent.epsilon = epsilon
    
    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = tuple(next_state)  # Convert next_state to tuple
        
        agent.update_q_table(state, action, reward, next_state)  # Update Q-table
        state = next_state
        total_reward += reward
        
        if done:
            if reward == -100:  # If agent hits obstacle, reset
                state = tuple(env.reset())
                total_reward = 0  # Reset total reward after crash
            break


    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
import pickle

# Save Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(agent.q_table, f)

print("Q-table saved successfully!")
