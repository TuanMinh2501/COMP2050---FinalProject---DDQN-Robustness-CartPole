# =====================================================================
# STATEMENT OF AUTHORSHIP & AI COLLABORATION LABELS
# =====================================================================
# [AI GENERATED] The standard Deep Q-Network forward mapping layers 
# and PyTorch neural layer setups inside 'DenseQEstimator' were generated 
# by AI assistant and modified for compliance.
#
# [OWN WORK] The specialized Reward Engineering function (`compute_custom_reward`),
# mathematical calculations for specific dynamic penalties (angle, position, velocity),
# and the hyperparameter selections were entirely engineered by the student.
# =====================================================================
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import copy

# =====================================================================
# NEURAL NETWORK ARCHITECTURE  (AI GENERATED & MODIFIED)
# =====================================================================

class DenseQEstimator(nn.Module):
    """
    Multi-layer Perceptron (MLP) mapping environmental states to action values.
    Uses a standard feed-forward architecture.
    """
    def __init__(self, input_features, total_actions):
        super(DenseQEstimator, self).__init__()
        
        # 3-layer architecture for stable state-space representation
        self.network_layers = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, total_actions)
        )
        
    def forward(self, state_tensor):
        # Computes expected Q-values for all possible actions
        return self.network_layers(state_tensor)

# =====================================================================
# CUSTOM DOUBLE DQN AGENT (OWN WORK)
# =====================================================================

class CustomDQNAgent:
    def __init__(self, state_size, action_size, variation_mode=False, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.variation_mode = variation_mode
        
        # Hyperparameters for training stability
        self.discount_factor = 0.99 
        self.exploration_rate = 1.0 
        self.decay_velocity = 0.995 
        self.min_exploration = 0.01 
        
        # Experience replay buffer to break correlation between samples
        self.memory_buffer = deque(maxlen=10000)
        
        # Initialize Online and Target networks for Double DQN
        self.brain_model = DenseQEstimator(state_size, action_size)
        self.target_brain_model = copy.deepcopy(self.brain_model)
        self.target_brain_model.eval() # Target model is not trained directly
        
        self.network_optimizer = optim.Adam(self.brain_model.parameters(), lr=0.001)

    def update_target_network(self):
        """Synchronize weights from online network to target network."""
        self.target_brain_model.load_state_dict(self.brain_model.state_dict())

    def select_action(self, current_state):
        """Epsilon-greedy action selection."""
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
            
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
        with torch.no_grad():
            predicted_q_values = self.brain_model(state_tensor)
        return torch.argmax(predicted_q_values).item()

    def store_transition(self, state, action, reward, next_state, execution_done):
        """Store experiences for batch training."""
        self.memory_buffer.append((state, action, reward, next_state, execution_done))

    def compute_custom_reward(self, current_state, baseline_reward):
        """
        [OWN WORK / THE "VIP" CORE]
        Calculates an engineered dual-shaping reward system based on physical constraints.
        Directly addresses systemic convergence latency by mapping continuous penalties 
        and strict balance hotspots.
        """
        if not self.variation_mode:
            return baseline_reward
            
        cart_pos, _, pole_angle, pole_vel = current_state
        
        # Define penalties and bonuses
        angle_penalty = -abs(pole_angle) * 3.0
        position_penalty = -abs(cart_pos) * 1.5
        velocity_penalty = -abs(pole_vel) * 0.5
        bonus = 2.0 if abs(cart_pos) < 0.1 and abs(pole_angle) < 0.05 else 0.0
            
        return baseline_reward + angle_penalty + position_penalty + velocity_penalty + bonus
    
    def train_network(self):
        """
        [AI HELP & MODIFIED] 
        Implements the standard Double DQN decoupling loss calculation.
        Modified by the student to support automated weight synchronization 
        via target model synchronization on every backpropagation step.
        """
        batch_size = 64
        if len(self.memory_buffer) < batch_size:
            return
            
        batch = random.sample(self.memory_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 1. Get current Q values from online network
        current_q = self.brain_model(states).gather(1, actions)
        
        # 2. Get target Q values (Double DQN decoupling)
        next_actions = self.brain_model(next_states).argmax(dim=1, keepdim=True)
        with torch.no_grad():
            next_q_values = self.target_brain_model(next_states).gather(1, next_actions)
            target_q = rewards + (self.discount_factor * next_q_values * (1 - dones))
        
        # 3. Compute loss and perform backpropagation
        loss = nn.MSELoss()(current_q, target_q)
        self.network_optimizer.zero_grad()
        loss.backward()
        self.network_optimizer.step()
        
        # Periodically update target network weights
        self.update_target_network()
        
        # Decay epsilon
        if self.exploration_rate > self.min_exploration:
            self.exploration_rate *= self.decay_velocity
