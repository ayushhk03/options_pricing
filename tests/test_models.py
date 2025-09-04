import pytest
import torch
from src.models.neural_network import OptionPricingNN, HestonNN


def test_option_pricing_nn():
    """Test OptionPricingNN model"""
    model = OptionPricingNN(input_dim=5)

    # Test forward pass
    x = torch.randn(10, 5)
    output = model(x)

    # Check output shape
    assert output.shape == (10, 1)

    # Check no NaN values
    assert not torch.isnan(output).any()


def test_heston_nn():
    """Test HestonNN model"""
    model = HestonNN(input_dim=9)

    # Test forward pass
    x = torch.randn(10, 9)
    output = model(x)

    # Check output shape
    assert output.shape == (10, 1)

    # Check no NaN values
    assert not torch.isnan(output).any()


if __name__ == "__main__":
    test_option_pricing_nn()
    test_heston_nn()
    print("All model tests passed!")