import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def heston_characteristic_function(phi, S0, K, T, r, v0, kappa, theta, sigma, rho):
    """
    Characteristic function for Heston model
    """
    # Parameters for characteristic function
    d = np.sqrt((rho * sigma * phi * 1j - kappa) ** 2 - sigma ** 2 * (1j * phi - phi ** 2))
    g = (kappa - rho * sigma * phi * 1j - d) / (kappa - rho * sigma * phi * 1j + d)

    # Characteristic function components
    C = (r * phi * 1j * T +
         (kappa * theta / sigma ** 2) *
         ((kappa - rho * sigma * phi * 1j - d) * T -
          2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))

    D = ((kappa - rho * sigma * phi * 1j - d) / sigma ** 2) * \
        ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))

    return np.exp(C + D * v0 + 1j * phi * np.log(S0))


def heston_integrand(phi, S0, K, T, r, v0, kappa, theta, sigma, rho, j):
    """
    Integrand for Heston option pricing formula
    """
    if j == 1:
        numerator = heston_characteristic_function(phi - 1j, S0, K, T, r, v0, kappa, theta, sigma, rho)
        denominator = 1j * phi * heston_characteristic_function(-1j, S0, K, T, r, v0, kappa, theta, sigma, rho)
    else:  # j == 2
        numerator = heston_characteristic_function(phi, S0, K, T, r, v0, kappa, theta, sigma, rho)
        denominator = 1j * phi

    return np.real(np.exp(-1j * phi * np.log(K)) * numerator / denominator)


def heston_price(S0, K, T, r, v0, kappa, theta, sigma, rho, option_type="call"):
    """
    Calculate Heston model option price using numerical integration

    Parameters:
    S0: Initial spot price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate
    v0: Initial variance
    kappa: Mean reversion rate
    theta: Long-run variance
    sigma: Volatility of volatility
    rho: Correlation between asset and volatility
    option_type: "call" or "put"

    Returns:
    Option price
    """
    # Numerical integration for P1 and P2
    P1 = 0.5 + (1 / np.pi) * quad(
        lambda phi: heston_integrand(phi, S0, K, T, r, v0, kappa, theta, sigma, rho, 1),
        0, 100, limit=1000
    )[0]

    P2 = 0.5 + (1 / np.pi) * quad(
        lambda phi: heston_integrand(phi, S0, K, T, r, v0, kappa, theta, sigma, rho, 2),
        0, 100, limit=1000
    )[0]

    if option_type == "call":
        price = S0 * P1 - K * np.exp(-r * T) * P2
    elif option_type == "put":
        price = K * np.exp(-r * T) * (1 - P2) - S0 * (1 - P1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def heston_price_approximate(S0, K, T, r, v0, kappa, theta, sigma, rho, option_type="call"):
    """
    Approximate Heston model price using closed-form approximation
    (Faster but less accurate than numerical integration)
    """
    # Simplified approximation - in practice, you'd use a more sophisticated method
    # This is a placeholder for demonstration
    avg_vol = np.sqrt(theta)
    d1 = (np.log(S0 / K) + (r + 0.5 * avg_vol ** 2) * T) / (avg_vol * np.sqrt(T))
    d2 = d1 - avg_vol * np.sqrt(T)

    if option_type == "call":
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    # Adjust for stochastic volatility (simplified)
    volatility_premium = 0.1 * sigma * np.sqrt(v0) * T
    if option_type == "call":
        price += volatility_premium
    else:
        price -= volatility_premium

    return max(price, 0)  # Ensure non-negative price


def generate_heston_dataset(n_samples=10000, S0_range=(50, 150), K_range=(50, 150),
                            T_range=(0.1, 2.0), r_range=(0.01, 0.1),
                            v0_range=(0.01, 0.2), kappa_range=(0.5, 2.0),
                            theta_range=(0.01, 0.2), sigma_range=(0.1, 0.4),
                            rho_range=(-0.9, 0.0)):
    """
    Generate dataset for Heston model training

    Parameters:
    n_samples: Number of samples to generate

    Returns:
    X: Input features [S0, K, T, r, v0, kappa, theta, sigma, rho]
    y: Option prices
    """
    np.random.seed(42)

    # Generate random parameters
    S0 = np.random.uniform(S0_range[0], S0_range[1], n_samples)
    K = np.random.uniform(K_range[0], K_range[1], n_samples)
    T = np.random.uniform(T_range[0], T_range[1], n_samples)
    r = np.random.uniform(r_range[0], r_range[1], n_samples)
    v0 = np.random.uniform(v0_range[0], v0_range[1], n_samples)
    kappa = np.random.uniform(kappa_range[0], kappa_range[1], n_samples)
    theta = np.random.uniform(theta_range[0], theta_range[1], n_samples)
    sigma = np.random.uniform(sigma_range[0], sigma_range[1], n_samples)
    rho = np.random.uniform(rho_range[0], rho_range[1], n_samples)

    # Calculate option prices using approximation (faster than full numerical integration)
    prices = np.zeros(n_samples)
    for i in range(n_samples):
        prices[i] = heston_price_approximate(
            S0[i], K[i], T[i], r[i], v0[i], kappa[i], theta[i], sigma[i], rho[i]
        )

    # Combine features
    X = np.column_stack([S0, K, T, r, v0, kappa, theta, sigma, rho])
    y = prices

    return X, y


def heston_greeks(S0, K, T, r, v0, kappa, theta, sigma, rho, option_type="call", method="finite_difference", h=0.001):
    """
    Calculate Greeks for Heston model using finite differences

    Parameters:
    h: Step size for finite differences

    Returns:
    Dictionary with delta, gamma, vega, etc.
    """
    # Calculate base price
    base_price = heston_price_approximate(S0, K, T, r, v0, kappa, theta, sigma, rho, option_type)

    # Delta (∂P/∂S)
    price_S_up = heston_price_approximate(S0 + h, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    price_S_down = heston_price_approximate(S0 - h, K, T, r, v0, kappa, theta, sigma, rho, option_type)
    delta = (price_S_up - price_S_down) / (2 * h)

    # Gamma (∂²P/∂S²)
    gamma = (price_S_up - 2 * base_price + price_S_down) / (h ** 2)

    # Vega (∂P/∂σ) - using volatility of volatility parameter
    price_sigma_up = heston_price_approximate(S0, K, T, r, v0, kappa, theta, sigma + h, rho, option_type)
    price_sigma_down = heston_price_approximate(S0, K, T, r, v0, kappa, theta, sigma - h, rho, option_type)
    vega = (price_sigma_up - price_sigma_down) / (2 * h)

    return {
        'price': base_price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega
    }