import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Define grid size
GRID_SIZE = 5

# Initialize Streamlit UI
st.title("Q-Learning in a Customizable 5x5 Grid Environment")
st.write("This is a simple environment where you can customize obstacles and run Q-Learning to train an agent to find the optimal path from start (1x1) to goal (5x5).")

# Initialize grid and obstacle setup
if 'grid' not in st.session_state:
    st.session_state['grid'] = np.zeros((GRID_SIZE, GRID_SIZE))
    st.session_state['start'] = (0, 0)  # Start point
    st.session_state['goal'] = (4, 4)   # Goal point
    st.session_state['obstacles'] = set()

# Define action space
actions = ['up', 'down', 'left', 'right']
action_to_index = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

# Define function to reset the environment
def reset_grid():
    st.session_state['grid'] = np.zeros((GRID_SIZE, GRID_SIZE))
    st.session_state['grid'][0, 0] = 1  # Start position
    st.session_state['grid'][4, 4] = 9  # Goal position

# Define function to add/remove obstacle
def toggle_obstacle(row, col):
    if (row, col) not in st.session_state['obstacles']:
        st.session_state['obstacles'].add((row, col))
        st.session_state['grid'][row, col] = -1  # Mark as obstacle
    else:
        st.session_state['obstacles'].remove((row, col))
        st.session_state['grid'][row, col] = 0  # Remove obstacle

# Display grid and interactive controls
st.write("Click on the grid cells to place/remove obstacles.")
for row in range(GRID_SIZE):
    cols = st.columns(GRID_SIZE)
    for col in range(GRID_SIZE):
        if (row, col) == st.session_state['start']:
            button_label = "S"
        elif (row, col) == st.session_state['goal']:
            button_label = "G"
        elif (row, col) in st.session_state['obstacles']:
            button_label = "X"
        else:
            button_label = ""
        
        if cols[col].button(button_label, key=f"{row}-{col}"):
            if (row, col) != st.session_state['start'] and (row, col) != st.session_state['goal']:
                toggle_obstacle(row, col)

# Define the reward matrix
reward_matrix = np.full((GRID_SIZE, GRID_SIZE), -1.0)  # Default reward for each step
reward_matrix[4, 4] = 100  # Goal reward

# Define Q-Learning algorithm
def q_learning(alpha=0.1, gamma=0.9, epsilon=0.9, episodes=1000):
    q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(actions)))
    for episode in range(episodes):
        state = st.session_state['start']
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)
            else:
                action = actions[np.argmax(q_table[state[0], state[1]])]
            
            next_state = get_next_state(state, action)
            reward = reward_matrix[next_state]
            
            if next_state == st.session_state['goal']:
                done = True
            
            # Q-Learning update
            current_q = q_table[state[0], state[1], action_to_index[action]]
            max_future_q = np.max(q_table[next_state[0], next_state[1]])
            new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
            q_table[state[0], state[1], action_to_index[action]] = new_q
            
            state = next_state
            
    return q_table

# Define function to get next state
def get_next_state(state, action):
    row, col = state
    if action == 'up' and row > 0:
        row -= 1
    elif action == 'down' and row < GRID_SIZE - 1:
        row += 1
    elif action == 'left' and col > 0:
        col -= 1
    elif action == 'right' and col < GRID_SIZE - 1:
        col += 1
    return (row, col)

# Function to visualize the agent's movement step-by-step
def visualize_agent_movement(q_table):
    state = st.session_state['start']
    path = [state]
    
    st.write("Visualizing agent's path from start to goal...")
    
    for _ in range(100):  # Avoid infinite loops by limiting to 100 steps
        action = actions[np.argmax(q_table[state[0], state[1]])]
        next_state = get_next_state(state, action)
        
        path.append(next_state)
        state = next_state

        if state == st.session_state['goal']:
            break
    
    return path

# Customization for hyperparameters
st.sidebar.header("Hyperparameters")
alpha = st.sidebar.slider("Learning Rate (alpha)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (gamma)", 0.01, 1.0, 0.9)
epsilon = st.sidebar.slider("Exploration Rate (epsilon)", 0.01, 1.0, 0.9)
episodes = st.sidebar.number_input("Number of Episodes", min_value=100, max_value=10000, value=1000)

# Run Q-Learning and show results
if st.sidebar.button("Run Q-Learning"):
    q_table = q_learning(alpha, gamma, epsilon, episodes)
    st.write("Q-Learning completed! You can inspect the learned Q-values below.")

    # Visualize the maximum Q-values (best action) for each state
    max_q_values = np.max(q_table, axis=2)  # Take max Q-value for each state

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(max_q_values, cmap='Blues')
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            ax.text(j, i, f'{max_q_values[i, j]:.2f}', ha='center', va='center', color='black')
    
    plt.xticks(np.arange(GRID_SIZE))
    plt.yticks(np.arange(GRID_SIZE))
    plt.grid(True)
    plt.title('Maximum Q-Values for Each State')
    st.pyplot(fig)
    
    # Visualize the agent's movement in real-time
    path = visualize_agent_movement(q_table)
    
    # Display each step of the agent's path
    for step, (row, col) in enumerate(path):
        st.write(f"Step {step + 1}: Agent moved to position ({row + 1}, {col + 1})")
        # Update grid visualization with the agent's current position
        temp_grid = np.copy(st.session_state['grid'])
        temp_grid[row, col] = 2  # Mark the agent's current position

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(temp_grid, cmap='coolwarm', interpolation='none')
        ax.set_xticks(np.arange(GRID_SIZE))
        ax.set_yticks(np.arange(GRID_SIZE))
        ax.grid(True)
        st.pyplot(fig)
        
        # Add a small delay to simulate real-time movement
        time.sleep(0.5)
