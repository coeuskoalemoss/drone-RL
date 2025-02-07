import numpy as np
from drone_env import DroneEnv
from agent import QLearningAgent

# Initialize environment and agent
env = DroneEnv()
agent = QLearningAgent(action_space=env.action_space, state_space=env.observation_space.shape, grid_size=env.grid_size)


# Hyperparameters
episodes = 1000
max_steps = 100

# Training loop
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        
        state = next_state
        total_reward += reward
        
        if done:
            break

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")
