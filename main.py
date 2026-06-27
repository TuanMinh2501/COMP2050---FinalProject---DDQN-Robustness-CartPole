import gymnasium as gym
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dqn_agent import CustomDQNAgent

# =====================================================================
# EXPERIMENTAL SETUP
# =====================================================================
TOTAL_SEEDS = 5
TRAINING_EPISODES = 120

def compute_moving_average(data_array, window=10):
    """Calculates smoothed trend lines to filter out RL noise."""
    return np.convolve(data_array, np.ones(window)/window, mode='valid')

def run_evaluation(variation_enabled):
    all_seeds_history = []
    global_best_score = 0
    
    for seed in range(TOTAL_SEEDS):
        print(f"--- Booting Seed {seed+1}/{TOTAL_SEEDS} (Custom Reward: {variation_enabled}) ---")
        
        # Initialize CartPole environment with fixed seed for reproducibility
        env = gym.make("CartPole-v1")
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # Instantiate our custom agent
        agent = CustomDQNAgent(state_size, action_size, variation_mode=variation_enabled)
        episode_scores = []
        
        for episode in range(TRAINING_EPISODES):
            current_state, _ = env.reset(seed=seed)
            total_score = 0
            execution_done = False
            with open("assets/training_log.csv", "a") as f:
                f.write(f"{seed},{episode},{total_score}\n")
            
            while not execution_done:
                # 1. Agent observes and acts
                action = agent.select_action(current_state)
                
                # 2. Environment reacts
                next_state, default_reward, terminated, truncated, _ = env.step(action)
                execution_done = terminated or truncated
                
                # 3. Apply our custom engineered reward (The "Vjp" core)
                # Note: We still track 'default_reward' for fair graphing
                engineered_reward = agent.compute_custom_reward(current_state, default_reward)
                
                # 4. Agent learns from the consequences
                agent.store_transition(current_state, action, engineered_reward, next_state, execution_done)
                
                # Trigger neural network backpropagation if memory has enough samples
                if len(agent.memory_buffer) > 64:
                    agent.train_network() # Note: We need to add this method to dqn_agent.py
                
                current_state = next_state
                total_score += default_reward
                
            episode_scores.append(total_score)
            
            # --- THE "VIP" FEATURE: SAVE BEST MODEL ---
            if variation_enabled and total_score > global_best_score:
                global_best_score = total_score
                torch.save(agent.brain_model.state_dict(), 'best_model.pth')
                
        env.close()
        all_seeds_history.append(episode_scores)
        
    return np.array(all_seeds_history)

if __name__ == "__main__":
    print("Initiating Baseline Evaluation...")
    baseline_data = run_evaluation(variation_enabled=False)
    
    print("\nInitiating Custom Dual-Shaping Evaluation...")
    custom_data = run_evaluation(variation_enabled=True)

    np.save('baseline_data.npy', baseline_data)
    np.save('custom_data.npy', custom_data)
    print("\n[SUCCESS] Raw matrix data exported to 'baseline_data.npy' and 'custom_data.npy'.")
