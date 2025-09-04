import pytest
import torch
from src.models.neural_network import OptionPricingNN
from src.greeks.autograd_greeks import get_delta, get_gamma, get_vega


def test_greeks_calculation():
    """Test Greek calculation functions"""
    model = OptionPricingNN(input_dim=5)

    # Create sample input (S, K, T, r, sigma)
    sample_input = torch.tensor([
        [100.0, 100.0, 1.0, 0.05, 0.2],
        [110.0, 100.0, 1.0, 0.05, 0.2],
    ], dtype=torch.float32)

    # Test delta calculation
    delta = get_delta(model, sample_input)
    assert delta.shape == (2,)

    # Test gamma calculation
    gamma = get_gamma(model, sample_input)
    assert gamma.shape == (2,)

    # Test vega calculation
    vega = get_vega(model, sample_input)
    assert vega.shape == (2,)

    # Check no NaN values
    assert not torch.isnan(delta).any()
    assert not torch.isnan(gamma).any()
    assert not torch.isnan(vega).any()


if __name__ == "__main__":
    test_greeks_calculation()
    print("All Greeks tests passed!")