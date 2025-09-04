import numpy as np
import torch


def finite_difference_greeks(model, inputs, h=0.001):
    """
    Calculate Greeks using finite difference method

    Parameters:
    model: Trained neural network model
    inputs: Input tensor [S, K, T, r, sigma] or [S, K, T, r, v0, kappa, theta, sigma, rho]
    h: Step size for finite differences

    Returns:
    Dictionary with price, delta, gamma, vega, theta, rho
    """
    # Ensure inputs require gradients
    inputs = inputs.clone().detach().requires_grad_(False)

    # Base price
    with torch.no_grad():
        base_price = model(inputs).item()

    # Initialize Greeks dictionary
    greeks = {
        'price': base_price,
        'delta': 0.0,
        'gamma': 0.0,
        'vega': 0.0,
        'theta': 0.0,
        'rho': 0.0
    }

    # Calculate Delta (∂P/∂S)
    inputs_up = inputs.clone()
    inputs_down = inputs.clone()
    inputs_up[0, 0] += h  # S + h
    inputs_down[0, 0] -= h  # S - h

    with torch.no_grad():
        price_S_up = model(inputs_up).item()
        price_S_down = model(inputs_down).item()

    greeks['delta'] = (price_S_up - price_S_down) / (2 * h)
    greeks['gamma'] = (price_S_up - 2 * base_price + price_S_down) / (h ** 2)

    # Calculate Vega (∂P/∂σ)
    # For Black-Scholes: sigma is at index 4
    # For Heston: sigma is at index 7
    if inputs.shape[1] == 5:  # Black-Scholes
        sigma_idx = 4
    else:  # Heston
        sigma_idx = 7

    inputs_sigma_up = inputs.clone()
    inputs_sigma_down = inputs.clone()
    inputs_sigma_up[0, sigma_idx] += h
    inputs_sigma_down[0, sigma_idx] -= h

    with torch.no_grad():
        price_sigma_up = model(inputs_sigma_up).item()
        price_sigma_down = model(inputs_sigma_down).item()

    greeks['vega'] = (price_sigma_up - price_sigma_down) / (2 * h)

    # Calculate Theta (∂P/∂T)
    inputs_T_up = inputs.clone()
    inputs_T_down = inputs.clone()
    inputs_T_up[0, 2] += h  # T + h
    inputs_T_down[0, 2] -= h  # T - h

    with torch.no_grad():
        price_T_up = model(inputs_T_up).item()
        price_T_down = model(inputs_T_down).item()

    greeks['theta'] = (price_T_up - price_T_down) / (2 * h)

    # Calculate Rho (∂P/∂r)
    inputs_r_up = inputs.clone()
    inputs_r_down = inputs.clone()
    inputs_r_up[0, 3] += h  # r + h
    inputs_r_down[0, 3] -= h  # r - h

    with torch.no_grad():
        price_r_up = model(inputs_r_up).item()
        price_r_down = model(inputs_r_down).item()

    greeks['rho'] = (price_r_up - price_r_down) / (2 * h)

    return greeks


def compare_greeks_methods(model, inputs, h=0.001):
    """
    Compare autograd and finite difference methods for Greek calculation

    Returns:
    Dictionary with comparison results
    """
    from .autograd_greeks import get_delta, get_gamma, get_vega

    # Finite difference Greeks
    fd_greeks = finite_difference_greeks(model, inputs, h)

    # Autograd Greeks
    delta_ag = get_delta(model, inputs).item()
    gamma_ag = get_gamma(model, inputs).item()
    vega_ag = get_vega(model, inputs).item()

    comparison = {
        'price': fd_greeks['price'],
        'delta': {
            'autograd': delta_ag,
            'finite_difference': fd_greeks['delta'],
            'difference': abs(delta_ag - fd_greeks['delta'])
        },
        'gamma': {
            'autograd': gamma_ag,
            'finite_difference': fd_greeks['gamma'],
            'difference': abs(gamma_ag - fd_greeks['gamma'])
        },
        'vega': {
            'autograd': vega_ag,
            'finite_difference': fd_greeks['vega'],
            'difference': abs(vega_ag - fd_greeks['vega'])
        }
    }

    return comparison