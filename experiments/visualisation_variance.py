import matplotlib.pyplot as plt
import numpy as np

def compute_moving_average(data_array, window=10):
    """Calculates smoothed trend lines to filter out RL noise."""
    return np.convolve(data_array, np.ones(window)/window, mode='valid')

def plot_scientific_results():
    """
    [OWN WORK] Loads training matrices, computes mathematical variance (std deviation),
    and plots professional scientific curves with shaded variance boundaries.
    """
    try:
        # Load the binary numpy matrices exported from main.py
        baseline_data = np.load('baseline_data.npy')
        custom_data = np.load('custom_data.npy')
    except FileNotFoundError:
        print("[ERROR] Matrix logs not found. Please run main.py completely first.")
        return

    # 1. Calculate statistical metrics across all seeds (axis=0)
    base_mean = np.mean(baseline_data, axis=0)
    base_std = np.std(baseline_data, axis=0)
    
    custom_mean = np.mean(custom_data, axis=0)
    custom_std = np.std(custom_data, axis=0)
    
    # 2. Apply moving average filter to BOTH mean values and variance boundaries
    smooth_base_mean = compute_moving_average(base_mean)
    smooth_base_std = compute_moving_average(base_std)
    
    smooth_custom_mean = compute_moving_average(custom_mean)
    smooth_custom_std = compute_moving_average(custom_std)
    
    # Generate identical x-axis matching the valid convolution zone
    window_offset = 10 - 1
    x_axis = np.arange(len(smooth_base_mean)) + window_offset + 1

    # 3. Setup professional figure aesthetics
    plt.figure(figsize=(11, 6))
    
    # Plot Baseline Trends (Mean line + Shaded Variance)
    plt.plot(x_axis, smooth_base_mean, label='Baseline DQN (Mean)', color='#3498db', linewidth=2)
    plt.fill_between(x_axis, 
                     smooth_base_mean - smooth_base_std, 
                     smooth_base_mean + smooth_base_std, 
                     color='#3498db', alpha=0.15, label='Baseline Variance (1 Std Dev)')

    # Plot Custom Dual-Shaping Trends (Mean line + Shaded Variance)
    plt.plot(x_axis, smooth_custom_mean, label='Dual-Shaping DQN (Mean)', color='#e74c3c', linewidth=2.5)
    plt.fill_between(x_axis, 
                     smooth_custom_mean - smooth_custom_std, 
                     smooth_custom_mean + smooth_custom_std, 
                     color='#e74c3c', alpha=0.15, label='Dual-Shaping Variance (1 Std Dev)')

    # 4. Academic Chart Styling
    plt.title('Performance and Stability Evaluation: Baseline vs Dual-Shaping DQN', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Training Episodes', fontsize=11)
    plt.ylabel('Cumulative Environmental Reward', fontsize=11)
    plt.xlim(window_offset + 1, len(base_mean))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', frameon=True, shadow=False, facecolor='white', edgecolor='#e2e2e2')
    
    plt.tight_layout()
    plt.savefig('scientific_results_with_variance.png', dpi=300)
    print("[SUCCESS] High-resolution graph saved as 'scientific_results_with_variance.png'")
    plt.show()

if __name__ == "__main__":
    plot_scientific_results()