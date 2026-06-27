DDQN Robustness Analysis: CartPole-v1
This project explores the trade-off between learning efficiency and policy robustness in reinforcement learning. By implementing a Double Deep Q-Network (DDQN) with a custom Dual-Reward Shaping mechanism, this project demonstrates how aggressive reward optimization can lead to "environmental overfitting" in the CartPole-v1 environment.

Key Features
Dual-Shaping DDQN: Enhanced learning stability through a custom reward shaping mechanism.

Robustness Testing: A custom testing framework that subjects the agent to stochastic lateral wind disturbances.

Real-time Visualization: A dedicated showcase using Pygame to visualize the agent's performance and failure modes in real-time.

Project Structure
Plaintext


/
├── assets/             # Pre-trained models and result plots
├── data/               # Experimental results (.npy files)
├── demo/               # Showcase scripts
├── experiments/        # Research scripts and visualizations
├── src/                # Core agent implementation
├── main.py             # Main entry point for training
└── requirements.txt    # Required dependencies

Getting Started
Prerequisites
Ensure you have Python installed, then install the required dependencies:

Bash


pip install gymnasium torch numpy pygame
Running the Showcase
To observe the agent's behavior under wind disturbances, navigate to the project root and run:

Bash


python demo/showcase.py
Controls: Press SPACE to pause/resume the simulation.

Research Findings
My analysis reveals that while dense reward shaping accelerates convergence in nominal conditions, it creates a rigid policy that lacks the recovery mechanics necessary for unpredictable environments. The "environmental overfitting" effect observed suggests that future research should prioritize domain randomization to improve agent adaptability.

License
This project is licensed under the MIT License.