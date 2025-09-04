import torch
from src.data_generation.monte_carlo import generate_geometric_brownian_motion, european_option_price
from src.models.neural_network import OptionPricingNN
from src.models.training import train_model, generate_dataset
from src.greeks.autograd_greeks import get_delta, get_gamma, get_vega
from src.utils.config import BLACK_SCHOLES_PARAMS, TRAINING_PARAMS, MC_PARAMS


def main():
    # Generate Monte Carlo data for benchmarking
    print("Generating Monte Carlo data...")
    paths = generate_geometric_brownian_motion(
        S0=BLACK_SCHOLES_PARAMS['S0'],
        r=BLACK_SCHOLES_PARAMS['r'],
        sigma=BLACK_SCHOLES_PARAMS['sigma'],
        T=BLACK_SCHOLES_PARAMS['T'],
        steps=MC_PARAMS['n_steps'],
        n_paths=MC_PARAMS['n_paths']
    )

    mc_price = european_option_price(
        paths,
        K=BLACK_SCHOLES_PARAMS['K'],
        r=BLACK_SCHOLES_PARAMS['r'],
        T=BLACK_SCHOLES_PARAMS['T']
    )

    print(f"Monte Carlo price: {mc_price:.4f}")

    # Generate training data
    print("Generating training data...")
    train_inputs, train_targets = generate_dataset("black_scholes", n_samples=100000)

    # Create data loaders
    dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAINING_PARAMS['batch_size'], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=TRAINING_PARAMS['batch_size'], shuffle=False
    )

    # Initialize and train model
    print("Training neural network...")
    model = OptionPricingNN(input_dim=5, hidden_layers=TRAINING_PARAMS['hidden_layers'])
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=TRAINING_PARAMS['num_epochs'],
        learning_rate=TRAINING_PARAMS['learning_rate']
    )

    # Calculate Greeks
    print("Calculating Greeks...")
    # Create sample input (S, K, T, r, sigma)
    sample_input = torch.tensor([
        BLACK_SCHOLES_PARAMS['S0'],
        BLACK_SCHOLES_PARAMS['K'],
        BLACK_SCHOLES_PARAMS['T'],
        BLACK_SCHOLES_PARAMS['r'],
        BLACK_SCHOLES_PARAMS['sigma']
    ], dtype=torch.float32).unsqueeze(0)

    delta = get_delta(model, sample_input)
    gamma = get_gamma(model, sample_input)
    vega = get_vega(model, sample_input)

    print(f"NN Delta: {delta.item():.4f}")
    print(f"NN Gamma: {gamma.item():.6f}")
    print(f"NN Vega: {vega.item():.4f}")

    # Compare performance
    import time

    # Time Monte Carlo
    start_time = time.time()
    for _ in range(100):
        paths = generate_geometric_brownian_motion(
            S0=BLACK_SCHOLES_PARAMS['S0'],
            r=BLACK_SCHOLES_PARAMS['r'],
            sigma=BLACK_SCHOLES_PARAMS['sigma'],
            T=BLACK_SCHOLES_PARAMS['T'],
            steps=MC_PARAMS['n_steps'],
            n_paths=1000  # Smaller for speed
        )
        mc_price = european_option_price(
            paths,
            K=BLACK_SCHOLES_PARAMS['K'],
            r=BLACK_SCHOLES_PARAMS['r'],
            T=BLACK_SCHOLES_PARAMS['T']
        )
    mc_time = time.time() - start_time

    # Time neural network
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            nn_price = model(sample_input)
    nn_time = time.time() - start_time

    print(f"Monte Carlo time for 100 prices: {mc_time:.4f}s")
    print(f"Neural network time for 100 prices: {nn_time:.6f}s")
    print(f"Speedup: {mc_time / nn_time:.1f}x")


if __name__ == "__main__":
    main()