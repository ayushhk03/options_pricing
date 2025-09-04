import numpy as np
import torch


def generate_geometric_brownian_motion(S0, r, sigma, T, steps, n_paths):
    """
    Generate Geometric Brownian Motion paths for Black-Scholes model
    """
    dt = T / steps
    paths = np.zeros((steps + 1, n_paths))
    paths[0] = S0

    for t in range(1, steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    return paths


def generate_heston_paths(S0, r, v0, kappa, theta, sigma, rho, T, steps, n_paths):
    """
    Generate paths using Heston model
    """
    dt = T / steps
    size = (n_paths, steps)
    prices = np.zeros((steps + 1, n_paths))
    vols = np.zeros((steps + 1, n_paths))

    prices[0] = S0
    vols[0] = v0

    for t in range(1, steps + 1):
        # Correlated random shocks
        z1 = np.random.standard_normal(n_paths)
        z2 = np.random.standard_normal(n_paths)
        z_v = z1
        z_s = rho * z1 + np.sqrt(1 - rho ** 2) * z2

        # Update volatility (ensure it stays positive)
        vols[t] = np.maximum(vols[t - 1] + kappa * (theta - vols[t - 1]) * dt +
                             sigma * np.sqrt(vols[t - 1] * dt) * z_v, 0)

        # Update price
        prices[t] = prices[t - 1] * np.exp((r - 0.5 * vols[t - 1]) * dt +
                                           np.sqrt(vols[t - 1] * dt) * z_s)

    return prices, vols


def european_option_price(paths, K, r, T, option_type="call"):
    """
    Calculate European option price from Monte Carlo paths
    """
    if option_type == "call":
        payoffs = np.maximum(paths[-1] - K, 0)
    else:  # put
        payoffs = np.maximum(K - paths[-1], 0)

    return np.exp(-r * T) * np.mean(payoffs)