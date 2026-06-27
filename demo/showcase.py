import gymnasium as gym
import torch
import numpy as np
import pygame # type: ignore
import random
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dqn_agent import CustomDQNAgent
# ==============================================================================================
# SHOWCASE DEMONSTRATION: ROBUSTNESS TEST WITH WIND DISTURBANCE (OWN WORK WITH SOME AI MODIFIED)
# ==============================================================================================
def run_showcase():
    # Configuration
    SHOWCASE_SPEED = 0.08      # Increase for slower simulation
    WIND_ALERT_DURATION = 30   # Frames of red warning
    
    print("Loading the trained brain (best_model.pth) for Robustness Test...")
    
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Instantiate agent
    agent = CustomDQNAgent(state_size, action_size, variation_mode=True)
    agent.brain_model.load_state_dict(torch.load('assets/best_model.pth', weights_only=True))
    agent.brain_model.eval()
    agent.exploration_rate = 0.0
    
    pygame.init()
    screen_width, screen_height = 600, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("COMP2050 - Robustness Test: DDQN vs Wind")
    clock = pygame.time.Clock()
    
    current_state, _ = env.reset()
    execution_done = False
    total_score = 0
    step_count = 0
    wind_active = False
    wind_duration = 0
    wind_direction = 0

    # Colors
    COLOR_BG_NORMAL = (18, 18, 18)
    COLOR_BG_WIND = (60, 20, 20) 
    COLOR_CART = (0, 188, 212)
    COLOR_POLE = (255, 152, 0)
    COLOR_AXLE = (158, 158, 158)
    COLOR_TRACK = (60, 60, 60)

    def draw_environment(state, surface, is_windy):
        surface.fill(COLOR_BG_WIND if is_windy else COLOR_BG_NORMAL)
        pygame.draw.line(surface, COLOR_TRACK, (0, 300), (600, 300), 2)
        
        cart_x = int(state[0] * 120 + 300)
        cart_y = 300
        
        # Draw Cart and Pole
        pygame.draw.rect(surface, COLOR_CART, (cart_x - 40, cart_y - 20, 80, 40), border_radius=5)
        pygame.draw.circle(surface, COLOR_AXLE, (cart_x, cart_y), 8)
        
        pole_end_x = cart_x + 150 * np.sin(state[2])
        pole_end_y = cart_y - 150 * np.cos(state[2])
        pygame.draw.line(surface, COLOR_POLE, (cart_x, cart_y), (int(pole_end_x), int(pole_end_y)), 10)
        
        if is_windy:
            font = pygame.font.SysFont("Arial", 28, bold=True)
            text = font.render("WIND DISTURBANCE!", True, (255, 255, 255))
            surface.blit(text, (130, 50))
            
        pygame.display.flip()

    # Main Loop
    while not execution_done:
        # Handle Events (Quit & Pause)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                execution_done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = True
                while paused:
                    for e in pygame.event.get():
                        if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                            paused = False
                        if e.type == pygame.QUIT:
                            paused = False
                            execution_done = True

        time.sleep(SHOWCASE_SPEED)
        step_count += 1
        
        # Wind Logic
        if not wind_active and step_count % random.randint(50, 70) == 0:
            wind_active = True
            wind_duration = WIND_ALERT_DURATION
            wind_direction = random.choice([-1, 1])

        if wind_active:
            current_state[1] += wind_direction * 0.4
            current_state[3] += wind_direction * 0.1
            wind_duration -= 1
            if wind_duration <= 0: wind_active = False

        # AI Decision
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            action = torch.argmax(agent.brain_model(state_tensor)).item()
            
        current_state, reward, terminated, truncated, _ = env.step(action)
        execution_done = terminated or truncated
        total_score += reward
        
        draw_environment(current_state, screen, wind_active)
        clock.tick(60)
        
    print(f"\n[FINISH] Showcase ended. Score: {total_score}")
    pygame.quit()
    env.close()

if __name__ == "__main__":
    run_showcase()