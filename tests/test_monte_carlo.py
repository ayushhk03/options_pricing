import pytest
import numpy as np
from src.data_generation.monte_carlo import generate_geometric_brownian_motion, european_option_price


def test_gbm_generation():
    """Test Geometric Brownian Motion path generation"""
    paths = generate_geometric_brownian_motion(S0=100, r=0.05, sigma=0.2, T=1.0, steps=252, n_paths=1000)

    # Check shape
    assert paths.shape == (253, 1000)  # steps + 1, n_paths

    # Check initial value
    assert np.allclose(paths[0], 100)

    # Check no negative values
    assert np.all(paths >= 0)


def test_option_pricing():
    """Test European option pricing"""
    paths = generate_geometric_brownian_motion(S0=100, r=0.05, sigma=0.2, T=1.0, steps=252, n_paths=10000)

    # Price a call option
    price = european_option_price(paths, K=100, r=0.05, T=1.0, option_type="call")

    # Price should be positive
    assert price > 0

    # Price should be reasonable (for ATM option with these parameters)
    assert 7 < price < 13


if __name__ == "__main__":
    test_gbm_generation()
    test_option_pricing()
    print("All Monte Carlo tests passed!")