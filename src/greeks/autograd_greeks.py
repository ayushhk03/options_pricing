import torch


def calculate_greeks(model, inputs, create_graph=False):
    """
    Calculate Greeks using PyTorch autograd
    """
    # Ensure we're in training mode to enable gradients
    model.train()

    # Convert inputs to require gradients
    inputs.requires_grad_(True)

    # Forward pass
    price = model(inputs)

    # Calculate first derivatives (Delta, Vega, etc.)
    first_derivatives = torch.autograd.grad(
        price, inputs,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
        retain_graph=True
    )[0]

    # Calculate second derivatives (Gamma)
    if create_graph:
        second_derivatives = []
        for i in range(inputs.size(1)):
            grad_i = torch.autograd.grad(
                first_derivatives[:, i], inputs,
                grad_outputs=torch.ones_like(first_derivatives[:, i]),
                create_graph=False,
                retain_graph=True
            )[0][:, i]
            second_derivatives.append(grad_i.unsqueeze(1))

        second_derivatives = torch.cat(second_derivatives, dim=1)
    else:
        second_derivatives = None

    return price, first_derivatives, second_derivatives


def get_delta(model, inputs):
    """
    Calculate Delta (∂Price/∂S)
    """
    price, first_derivatives, _ = calculate_greeks(model, inputs)
    return first_derivatives[:, 0]  # Assuming S is the first input


def get_gamma(model, inputs):
    """
    Calculate Gamma (∂²Price/∂S²)
    """
    _, _, second_derivatives = calculate_greeks(model, inputs, create_graph=True)
    return second_derivatives[:, 0]  # Assuming S is the first input


def get_vega(model, inputs):
    """
    Calculate Vega (∂Price/∂σ)
    """
    price, first_derivatives, _ = calculate_greeks(model, inputs)
    return first_derivatives[:, 4]  # Assuming sigma is the fifth input for Black-Scholes