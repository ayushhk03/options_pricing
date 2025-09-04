# Options Pricing using Neural Networks and Greeks Approximation

A high-performance options pricing system that uses neural networks to approximate Black-Scholes and Heston models with automatic Greek calculation using PyTorch's autograd.

## Features

- ⚡ **100x Faster Pricing**: Neural network inference outperforms Monte Carlo simulations
- 🎯 **<1% Pricing Error**: Accurate approximation of Black-Scholes and Heston models
- 📊 **Automatic Greeks Calculation**: Delta, Gamma, Vega using PyTorch autograd
- 🔄 **Multiple Models**: Support for both Black-Scholes and Heston stochastic volatility model
- 📈 **Comprehensive Benchmarking**: Performance comparison against traditional methods
- 🧪 **Full Test Suite**: Unit tests for all components

## Performance Highlights

- **100x speedup** compared to Monte Carlo simulations
- **<1% pricing error** versus ground truth Monte Carlo prices
- **Real-time Greek calculation** with automatic differentiation
- **Scalable architecture** for large-scale option portfolios

## Project Structure

```
options-pricing-nn/
├── src/
│   ├── data_generation/
│   │   ├── monte_carlo.py          # Monte Carlo simulation engines
│   │   ├── black_scholes.py        # Black-Scholes model implementation
│   │   └── heston_model.py         # Heston model implementation
│   ├── models/
│   │   ├── neural_network.py       # Neural network architectures
│   │   ├── training.py             # Model training utilities
│   │   └── evaluation.py           # Model evaluation metrics
│   ├── greeks/
│   │   ├── autograd_greeks.py      # Greek calculation using autograd
│   │   └── finite_difference.py    # Finite difference Greek calculation
│   └── utils/
│       ├── data_processing.py      # Data preprocessing utilities
│       ├── visualization.py        # Plotting and visualization
│       └── config.py               # Configuration parameters
├── notebooks/
│   ├── 01_data_generation.ipynb    # Generate training data
│   ├── 02_model_training.ipynb     # Train neural network models
│   ├── 03_greeks_calculation.ipynb # Calculate and visualize Greeks
│   └── 04_performance_benchmarking.ipynb # Benchmark against Monte Carlo
├── tests/
│   ├── test_monte_carlo.py         # Monte Carlo simulation tests
│   ├── test_models.py              # Model architecture tests
│   └── test_greeks.py              # Greek calculation tests
├── data/
│   ├── raw/                        # Raw Monte Carlo simulation data
│   ├── processed/                  # Processed training data
│   └── models/                     # Trained model weights
├── results/                        # Benchmark results and plots
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
├── run_experiment.py               # Main execution script
└── README.md                       # Project documentation
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd options-pricing-nn
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install in development mode** (optional):
```bash
pip install -e .
```

## Quick Start

### Run the complete experiment:
```bash
python run_experiment.py
```

### Train a specific model:
```python
from src.models.neural_network import OptionPricingNN
from src.models.training import train_model

model = OptionPricingNN(input_dim=5)
train_losses, val_losses = train_model(model, train_loader, val_loader)
```

### Calculate Greeks:
```python
from src.greeks.autograd_greeks import get_delta, get_gamma, get_vega

inputs = torch.tensor([100.0, 100.0, 1.0, 0.05, 0.2])  # S, K, T, r, sigma
delta = get_delta(model, inputs)
gamma = get_gamma(model, inputs)
vega = get_vega(model, inputs)
```

## Usage Examples

### Basic Option Pricing
```python
from src.data_generation.monte_carlo import generate_geometric_brownian_motion
from src.data_generation.monte_carlo import european_option_price

# Generate Monte Carlo paths
paths = generate_geometric_brownian_motion(
    S0=100, r=0.05, sigma=0.2, T=1.0, steps=252, n_paths=10000
)

# Price European call option
price = european_option_price(paths, K=100, r=0.05, T=1.0, option_type="call")
print(f"Option price: ${price:.2f}")
```

### Neural Network Pricing
```python
from src.models.neural_network import OptionPricingNN

# Load trained model
model = OptionPricingNN(input_dim=5)
model.load_state_dict(torch.load('data/models/option_pricing_nn.pth'))

# Price using neural network
inputs = torch.tensor([[100.0, 100.0, 1.0, 0.05, 0.2]])  # S, K, T, r, sigma
nn_price = model(inputs)
print(f"NN price: ${nn_price.item():.2f}")
```

## Notebook Workflow

1. **Data Generation** (`01_data_generation.ipynb`):
   - Generate Monte Carlo paths for training
   - Create Black-Scholes and Heston model datasets

2. **Model Training** (`02_model_training.ipynb`):
   - Train neural network models
   - Validate pricing accuracy
   - Save trained models

3. **Greeks Calculation** (`03_greeks_calculation.ipynb`):
   - Calculate Delta, Gamma, Vega using autograd
   - Visualize Greek surfaces
   - Compare with finite difference methods

4. **Performance Benchmarking** (`04_performance_benchmarking.ipynb`):
   - Compare speed with Monte Carlo
   - Measure pricing accuracy
   - Generate performance reports

## Running Tests

Run all tests:
```bash
python -m pytest tests/ -v
```

Run specific test files:
```bash
python tests/test_monte_carlo.py
python tests/test_models.py
python tests/test_greeks.py
```

## Model Architectures

### OptionPricingNN
- **Input**: [S, K, T, r, sigma]
- **Architecture**: Fully connected network with ReLU activations
- **Output**: Option price

### HestonNN
- **Input**: [S, K, T, r, v0, kappa, theta, sigma, rho]
- **Architecture**: Deep network with feature extraction
- **Output**: Option price under Heston model

## Supported Greeks

- **Delta** (Δ): ∂Price/∂S
- **Gamma** (Γ): ∂²Price/∂S²
- **Vega** (ν): ∂Price/∂σ
- **Theta** (Θ): ∂Price/∂t
- **Rho** (ρ): ∂Price/∂r

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Speedup | 100x | Neural network vs Monte Carlo |
| Pricing Error | <1% | RMSE compared to Monte Carlo |
| Inference Time | ~0.1ms | Per option pricing |
| Training Time | ~5min | On CPU, 100K samples |

## Dependencies

- **PyTorch**: Neural network framework and autograd
- **NumPy**: Numerical computations
- **SciPy**: Scientific computing and special functions
- **Matplotlib**: Visualization and plotting
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Data preprocessing and metrics
- **tqdm**: Progress bars for training

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{options_pricing_nn,
  title = {Options Pricing using Neural Networks and Greeks Approximation},
  author = {Ayush Kumar},
  year = {2025},
  url = {https://github.com/ayushhk03/options-pricing-nn}
}
```

## Support

For questions and support:
- Open an issue on GitHub
- Contact: ayushk745@gmail.com

## Acknowledgments

- Black-Scholes model derivation
- Heston stochastic volatility model
- PyTorch autograd for automatic differentiation
- Monte Carlo methods for option pricing

---

**Note**: This is a research prototype. Always validate results against established pricing methods before use in production environments.