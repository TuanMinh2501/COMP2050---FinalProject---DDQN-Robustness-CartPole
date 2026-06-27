import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dqn_agent import CustomDQNAgent

# =====================================================================
# HYPER-PARAMETER EXPERIMENT: LEARNING RATE
# =====================================================================

TRAINING_EPISODES = 120
MAX_STEPS = 500
SEEDS = 3 # Number of random seeds for statistical robustness
LEARNING_RATES = [0.01, 0.001, 0.0001]
COLORS = ['#2ecc71', '#e74c3c', '#f39c12'] # Green, Red, Orange

def run_lr_evaluation(lr):
    """Runs the Dual-Shaping agent with a specific learning rate."""
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    all_scores = []
    
    for seed in range(SEEDS):
        # Initialize environment and agent for each seed
        env.reset(seed=seed)
        np.random.seed(seed)
        
        # Instantiate the agent with the specified learning rate
        agent = CustomDQNAgent(state_size, action_size, variation_mode=True, learning_rate=lr)
        seed_scores = []
        
        for e in range(TRAINING_EPISODES):
            state, _ = env.reset()
            total_reward = 0
            
            for _ in range(MAX_STEPS):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Áp dụng Custom Reward
                shaped_reward = agent.compute_custom_reward(next_state, reward)
                agent.store_transition(state, action, shaped_reward, next_state, done)
                agent.train_network()
                
                state = next_state
                total_reward += reward
                if done:
                    break
                    
            seed_scores.append(total_reward)
            print(f"[LR: {lr}] Seed: {seed+1}/{SEEDS} | Episode: {e+1}/{TRAINING_EPISODES} | Score: {total_reward}")
            
        all_scores.append(seed_scores)
        
    env.close()
    return np.array(all_scores)

# =====================================================================
# Creating the Plotting Function (OWN WORK WITH SOME AI MODIFIED)
# =====================================================================

def plot_lr_results(results_dict):
    """Plots the comparison of different learning rates."""
    plt.figure(figsize=(10, 6))
    
    window = 10
    x_axis = np.arange(TRAINING_EPISODES - window + 1) + window
    
    for idx, (lr, data) in enumerate(results_dict.items()):
        mean_data = np.mean(data, axis=0)
        # Smoothing
        smoothed_mean = np.convolve(mean_data, np.ones(window)/window, mode='valid')
        
        plt.plot(x_axis, smoothed_mean, label=f'LR = {lr}', color=COLORS[idx], linewidth=2.5)

    plt.title('Hyper-parameter Impact: Learning Rate on Dual-Shaping DQN', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Training Episodes', fontsize=11)
    plt.ylabel('Cumulative Reward (Moving Avg)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e2e2e2')
    
    plt.tight_layout()
    plt.savefig('hyperparam_lr_results.png', dpi=300)
    print("\n[SUCCESS] Chart saved as 'hyperparam_lr_results.png'")
    plt.show()

if __name__ == "__main__":
    results = {}
    for lr in LEARNING_RATES:
        print(f"\n--- STARTING EVALUATION FOR LR = {lr} ---")
        results[lr] = run_lr_evaluation(lr)
        
    plot_lr_results(results)