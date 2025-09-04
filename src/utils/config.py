# Model parameters
BLACK_SCHOLES_PARAMS = {
    'S0': 100.0,      # Initial stock price
    'K': 100.0,       # Strike price
    'T': 1.0,         # Time to maturity (years)
    'r': 0.05,        # Risk-free rate
    'sigma': 0.2,     # Volatility
}

HESTON_PARAMS = {
    'S0': 100.0,      # Initial stock price
    'K': 100.0,       # Strike price
    'T': 1.0,         # Time to maturity (years)
    'r': 0.05,        # Risk-free rate
    'v0': 0.04,       # Initial variance
    'kappa': 1.0,     # Mean reversion rate
    'theta': 0.04,    # Long-run variance
    'sigma': 0.1,     # Vol of vol
    'rho': -0.7,      # Correlation
}

# Training parameters
TRAINING_PARAMS = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'hidden_layers': [128, 128, 64, 32],
}

# Monte Carlo parameters
MC_PARAMS = {
    'n_paths': 100000,
    'n_steps': 252,
}