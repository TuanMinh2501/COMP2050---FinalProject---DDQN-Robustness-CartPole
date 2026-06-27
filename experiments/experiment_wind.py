import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dqn_agent import CustomDQNAgent

# =====================================================================
# STATEMENT OF AUTHORSHIP & AI COLLABORATION LABELS
# =====================================================================
# [OWN WORK] The wind disturbance logic and the robustness evaluation 
# framework were conceptualized and implemented by the student, adapted 
# from the showcase demonstration.
# =====================================================================

TRAINING_EPISODES = 60 # Fast training phase before testing
TEST_EPISODES = 10     # Number of episodes to average the wind test
MAX_STEPS = 500

# Define wind intensities to be applied to cart velocity
WIND_CONDITIONS = {
    "No Wind": 0.0,
    "Medium Gale": 0.5,
    "Strong Storm": 1.2
}

def apply_environmental_disturbance(env, base_wind_strength):
    """
    [OWN WORK] Injects artificial wind into the physics engine.
    Directly modifies the underlying state array [cart_pos, cart_vel, pole_angle, pole_vel].
    """
    if env.unwrapped.state is not None and base_wind_strength > 0.0:
        # Calculate actual wind force with a random gust factor
        gust_noise = np.random.uniform(-0.2, 0.2)
        actual_wind = base_wind_strength + gust_noise
        
        # Extract current state and convert immutable tuple to a mutable list
        current_state = list(env.unwrapped.state)
        
        # Apply wind force to the Cart Velocity (index 1 in the state array)
        current_state[1] += actual_wind
        env.unwrapped.state = tuple(current_state)

def train_and_test_agent(variation_mode, wind_strength):
    """Trains an agent briefly, then tests its robustness against wind."""
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = CustomDQNAgent(state_size, action_size, variation_mode=variation_mode)
    
    # 1. Quick Training Phase (Ideal Conditions)
    for e in range(TRAINING_EPISODES):
        state, _ = env.reset()
        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            shaped_reward = agent.compute_custom_reward(next_state, reward)
            agent.store_transition(state, action, shaped_reward, next_state, done)
            agent.train_network()
            
            state = next_state
            if done: break

    # 2. Testing Phase (Wind Conditions Applied)
    agent.exploration_rate = 0.0 # Turn off exploration for pure testing
    survival_steps_list = []
    
    for _ in range(TEST_EPISODES):
        state, _ = env.reset()
        steps_survived = 0
        
        for _ in range(MAX_STEPS):
            # Apply the custom wind disturbance BEFORE the agent reacts
            apply_environmental_disturbance(env, wind_strength)
            
            # Agent chooses best action based on trained policy
            action = agent.select_action(state)
            next_state, _, terminated, truncated, _ = env.step(action)
            
            state = next_state
            steps_survived += 1
            
            if terminated or truncated:
                break
                
        survival_steps_list.append(steps_survived)
        
    env.close()
    return np.mean(survival_steps_list)

def plot_robustness_results(baseline_results, custom_results):
    """
    [AI GENERATED & MODIFIED] Plots a grouped bar chart to compare
    survival rates under varying physical disturbances.
    """
    labels = list(WIND_CONDITIONS.keys())
    x = np.arange(len(labels))
    width = 0.35  

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting Baseline vs Custom Dual-Shaping
    rects1 = ax.bar(x - width/2, baseline_results, width, label='Baseline DQN', color='#3498db')
    rects2 = ax.bar(x + width/2, custom_results, width, label='Dual-Shaping DQN', color='#e74c3c')

    ax.set_ylabel('Average Survival Steps (Max 500)', fontsize=12)
    ax.set_title('Robustness Evaluation: Agent Survival Under Wind Disturbances', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e2e2e2')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Attach a text label above each bar displaying its height
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

    fig.tight_layout()
    plt.savefig('robustness_wind_test.png', dpi=300)
    print("\n[SUCCESS] Bar chart saved as 'robustness_wind_test.png'")
    plt.show()

if __name__ == "__main__":
    baseline_survival = []
    custom_survival = []
    
    print("Starting Wind Robustness Evaluation... This will take a few minutes.")
    
    for condition_name, strength in WIND_CONDITIONS.items():
        print(f"\nEvaluating under {condition_name} (Force: {strength})...")
        
        print("  -> Training and testing Baseline Agent...")
        base_score = train_and_test_agent(variation_mode=False, wind_strength=strength)
        baseline_survival.append(base_score)
        
        print("  -> Training and testing Dual-Shaping Agent...")
        custom_score = train_and_test_agent(variation_mode=True, wind_strength=strength)
        custom_survival.append(custom_score)
        
        print(f"  [Result] Baseline: {base_score:.1f} steps | Dual-Shaping: {custom_score:.1f} steps")

    plot_robustness_results(baseline_survival, custom_survival)