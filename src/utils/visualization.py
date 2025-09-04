import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def setup_plot_style():
    """Set up consistent plot style"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def plot_training_history(train_losses, val_losses, model_name="Model"):
    """Plot training and validation loss history"""
    setup_plot_style()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{model_name}: Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.semilogy(train_losses, label='Training Loss')
    plt.semilogy(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log MSE Loss')
    plt.title(f'{model_name}: Log Scale Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    return plt


def plot_greek_surface(greek_values, S_range, param_range, greek_name="Delta", param_name="Volatility"):
    """Create 3D surface plot for Greek values"""
    setup_plot_style()

    S_mesh, param_mesh = np.meshgrid(S_range, param_range)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(S_mesh, param_mesh, greek_values,
                           cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True)

    ax.set_xlabel('Spot Price (S)')
    ax.set_ylabel(param_name)
    ax.set_zlabel(greek_name)
    ax.set_title(f'{greek_name} Surface: {greek_name} vs Spot Price and {param_name}')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    return plt


def plot_performance_comparison(mc_times, nn_times, mc_prices, nn_prices):
    """Plot performance comparison between Monte Carlo and Neural Network"""
    setup_plot_style()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Speed comparison
    speedup = np.array(mc_times) / np.array(nn_times)
    ax1.bar(['Monte Carlo', 'Neural Network'], [np.mean(mc_times), np.mean(nn_times)],
            yerr=[np.std(mc_times), np.std(nn_times)], capsize=5)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.text(0.5, 0.9, f'Speedup: {np.mean(speedup):.1f}x',
             transform=ax1.transAxes, ha='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    # Price comparison scatter
    ax2.scatter(mc_prices, nn_prices, alpha=0.6)
    max_price = max(max(mc_prices), max(nn_prices))
    ax2.plot([0, max_price], [0, max_price], 'r--', lw=2)
    ax2.set_xlabel('Monte Carlo Prices')
    ax2.set_ylabel('Neural Network Prices')
    ax2.set_title('Price Comparison')
    ax2.grid(True, alpha=0.3)

    # Error distribution
    errors = np.abs(np.array(mc_prices) - np.array(nn_prices))
    percentage_errors = (errors / np.array(mc_prices)) * 100
    ax3.hist(percentage_errors, bins=30, alpha=0.7, edgecolor='black')
    ax3.axvline(1.0, color='red', linestyle='--', label='1% Error')
    ax3.set_xlabel('Percentage Error (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Cumulative error distribution
    sorted_errors = np.sort(percentage_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax4.plot(sorted_errors, cumulative, marker='.', linestyle='none')
    ax4.set_xlabel('Percentage Error (%)')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Error Distribution')
    ax4.grid(True, alpha=0.3)

    # Add text with statistics
    stats_text = f"""Statistics:
Mean Error: {np.mean(percentage_errors):.2f}%
Max Error: {np.max(percentage_errors):.2f}%
Errors < 1%: {np.sum(percentage_errors < 1.0)}/{len(percentage_errors)}
({np.mean(percentage_errors < 1.0) * 100:.1f}%)"""

    ax4.text(0.95, 0.05, stats_text, transform=ax4.transAxes,
             ha='right', va='bottom', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    return plt


def save_plot(plt, filename, dpi=300):
    """Save plot to file"""
    plt.savefig(f'../results/{filename}', dpi=dpi, bbox_inches='tight')
    plt.close()