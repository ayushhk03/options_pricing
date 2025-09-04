import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, X_test, y_test, model_type="black_scholes"):
    """
    Evaluate model performance

    Parameters:
    model: Trained neural network
    X_test: Test features
    y_test: Test targets
    model_type: "black_scholes" or "heston"

    Returns:
    Dictionary with evaluation metrics
    """
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        y_pred = model(X_tensor).numpy()

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate percentage errors
    percentage_errors = np.abs((y_test.flatten() - y_pred.flatten()) / y_test.flatten()) * 100
    mean_percentage_error = np.mean(percentage_errors)
    max_percentage_error = np.max(percentage_errors)

    # Count errors less than 1%
    errors_less_than_1pct = np.sum(percentage_errors < 1.0)
    pct_errors_less_than_1pct = (errors_less_than_1pct / len(percentage_errors)) * 100

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mean_percentage_error': mean_percentage_error,
        'max_percentage_error': max_percentage_error,
        'errors_less_than_1pct': errors_less_than_1pct,
        'pct_errors_less_than_1pct': pct_errors_less_than_1pct,
        'predictions': y_pred,
        'targets': y_test
    }

    return metrics


def plot_predictions_vs_targets(metrics, model_name="Model"):
    """
    Create prediction vs target plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics['targets'], metrics['predictions'], alpha=0.5)

    # Perfect prediction line
    max_val = max(np.max(metrics['targets']), np.max(metrics['predictions']))
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2)

    plt.xlabel('True Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'{model_name}: Predictions vs True Prices\n'
              f'RMSE: {metrics["rmse"]:.4f}, R²: {metrics["r2"]:.4f}\n'
              f'Mean Error: {metrics["mean_percentage_error"]:.2f}%')
    plt.grid(True, alpha=0.3)

    return plt


def plot_error_distribution(metrics, model_name="Model"):
    """
    Plot distribution of percentage errors
    """
    percentage_errors = np.abs((metrics['targets'].flatten() - metrics['predictions'].flatten()) /
                               metrics['targets'].flatten()) * 100

    plt.figure(figsize=(10, 6))
    plt.hist(percentage_errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(1.0, color='red', linestyle='--', label='1% Error Threshold')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.title(f'{model_name}: Error Distribution\n'
              f'{metrics["pct_errors_less_than_1pct"]:.1f}% of errors < 1%')
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt


def benchmark_against_monte_carlo(model, monte_carlo_prices, nn_inputs, model_type="black_scholes"):
    """
    Benchmark neural network against Monte Carlo prices
    """
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(nn_inputs)
        nn_prices = model(X_tensor).numpy().flatten()

    # Calculate errors
    errors = np.abs(monte_carlo_prices - nn_prices)
    percentage_errors = (errors / monte_carlo_prices) * 100

    benchmark_metrics = {
        'monte_carlo_prices': monte_carlo_prices,
        'nn_prices': nn_prices,
        'mean_absolute_error': np.mean(errors),
        'max_absolute_error': np.max(errors),
        'mean_percentage_error': np.mean(percentage_errors),
        'max_percentage_error': np.max(percentage_errors),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'r2': r2_score(monte_carlo_prices, nn_prices)
    }

    return benchmark_metrics