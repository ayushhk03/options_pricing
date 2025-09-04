import numpy as np
from scipy.stats import norm


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes option price

    Parameters:
    S: Spot price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Volatility
    option_type: "call" or "put"

    Returns:
    Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def black_scholes_delta(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes delta

    Parameters:
    S: Spot price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Volatility
    option_type: "call" or "put"

    Returns:
    Delta (∂Price/∂S)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return delta


def black_scholes_gamma(S, K, T, r, sigma):
    """
    Calculate Black-Scholes gamma

    Parameters:
    S: Spot price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Volatility

    Returns:
    Gamma (∂²Price/∂S²)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma


def black_scholes_vega(S, K, T, r, sigma):
    """
    Calculate Black-Scholes vega

    Parameters:
    S: Spot price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Volatility

    Returns:
    Vega (∂Price/∂σ)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega


def black_scholes_theta(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes theta

    Parameters:
    S: Spot price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Volatility
    option_type: "call" or "put"

    Returns:
    Theta (∂Price/∂t)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "put":
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return theta


def black_scholes_rho(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes rho

    Parameters:
    S: Spot price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Volatility
    option_type: "call" or "put"

    Returns:
    Rho (∂Price/∂r)
    """
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return rho


def generate_black_scholes_dataset(n_samples=10000, S_range=(50, 150), K_range=(50, 150),
                                   T_range=(0.1, 2.0), r_range=(0.01, 0.1), sigma_range=(0.1, 0.5)):
    """
    Generate dataset for Black-Scholes model training

    Parameters:
    n_samples: Number of samples to generate
    S_range: Tuple of (min, max) for spot price
    K_range: Tuple of (min, max) for strike price
    T_range: Tuple of (min, max) for time to maturity
    r_range: Tuple of (min, max) for risk-free rate
    sigma_range: Tuple of (min, max) for volatility

    Returns:
    X: Input features [S, K, T, r, sigma]
    y: Option prices
    """
    np.random.seed(42)

    # Generate random parameters
    S = np.random.uniform(S_range[0], S_range[1], n_samples)
    K = np.random.uniform(K_range[0], K_range[1], n_samples)
    T = np.random.uniform(T_range[0], T_range[1], n_samples)
    r = np.random.uniform(r_range[0], r_range[1], n_samples)
    sigma = np.random.uniform(sigma_range[0], sigma_range[1], n_samples)

    # Calculate option prices
    prices = np.zeros(n_samples)
    for i in range(n_samples):
        prices[i] = black_scholes_price(S[i], K[i], T[i], r[i], sigma[i])

    # Combine features
    X = np.column_stack([S, K, T, r, sigma])
    y = prices

    return X, y