import sys
import os
import time
import streamlit as st
import torch

# --------------------------------------------------
# Path setup (NO Streamlit calls here)
# --------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

# --------------------------------------------------
# FIRST and ONLY Streamlit command
# --------------------------------------------------
st.set_page_config(
    page_title="Options Pricing Platform",
    layout="wide"
)

# --------------------------------------------------
# Imports from src/
# --------------------------------------------------
from data_generation.black_scholes import (
    black_scholes_price,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_vega,
    black_scholes_theta,
    black_scholes_rho
)

from data_generation.monte_carlo import (
    generate_geometric_brownian_motion,
    european_option_price
)

from models.neural_network import OptionPricingNN
from greeks.autograd_greeks import get_delta, get_gamma, get_vega

# --------------------------------------------------
# Load Neural Network (cached)
# --------------------------------------------------
@st.cache_resource
def load_bs_nn():
    model = OptionPricingNN(input_dim=5)
    model.load_state_dict(
        torch.load(
            os.path.join(ROOT_DIR, "data/models/option_pricing_nn.pth"),
            map_location="cpu"
        )
    )
    model.eval()
    return model

# --------------------------------------------------
# App UI
# --------------------------------------------------
st.title("Options Pricing Platform")
st.caption("Black–Scholes | Monte Carlo | Neural Network Surrogates")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Model Selection")

model = st.sidebar.radio(
    "Pricing Model",
    [
        "Black-Scholes",
        "Monte Carlo",
        "Neural Network (BS)"
    ]
)

st.sidebar.header("Option Parameters")

S = st.sidebar.number_input("Spot Price (S)", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (T, years)", value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)

# Monte Carlo specific
mc_paths = None
mc_steps = None
if model == "Monte Carlo":
    mc_paths = st.sidebar.number_input("Monte Carlo Paths", value=10000, step=1000)
    mc_steps = st.sidebar.number_input("Time Steps", value=252, step=10)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_price, tab_greeks, tab_benchmark = st.tabs(
    ["Pricing", "Greeks", "Benchmark"]
)

# --------------------------------------------------
# TAB 1 — Pricing
# --------------------------------------------------
with tab_price:
    st.subheader("Option Price")

    # Black–Scholes
    if model == "Black-Scholes":
        start = time.time()
        price = black_scholes_price(S, K, T, r, sigma)
        elapsed = (time.time() - start) * 1000

        st.metric("Option Price", f"{price:.4f}")
        st.caption(f"Inference Time: {elapsed:.3f} ms")

    # Monte Carlo
    elif model == "Monte Carlo":
        start = time.time()

        paths = generate_geometric_brownian_motion(
            S0=S,
            r=r,
            sigma=sigma,
            T=T,
            steps=int(mc_steps),
            n_paths=int(mc_paths)
        )

        price = european_option_price(
            paths=paths,
            K=K,
            r=r,
            T=T,
            option_type="call"
        )

        elapsed = (time.time() - start) * 1000

        st.metric("Option Price (Monte Carlo)", f"{price:.4f}")
        st.caption(f"Inference Time: {elapsed:.1f} ms")

    # Neural Network
    elif model == "Neural Network (BS)":
        nn_model = load_bs_nn()
        inputs = torch.tensor([[S, K, T, r, sigma]], dtype=torch.float32)

        start = time.time()
        with torch.no_grad():
            price = nn_model(inputs).item()
        elapsed = (time.time() - start) * 1000

        st.metric("Option Price (Neural Network)", f"{price:.4f}")
        st.caption(f"Inference Time: {elapsed:.3f} ms")

# --------------------------------------------------
# TAB 2 — Greeks
# --------------------------------------------------
with tab_greeks:
    st.subheader("Option Greeks")

    # Analytical Greeks
    if model == "Black-Scholes":
        st.table({
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Value": [
                black_scholes_delta(S, K, T, r, sigma),
                black_scholes_gamma(S, K, T, r, sigma),
                black_scholes_vega(S, K, T, r, sigma),
                black_scholes_theta(S, K, T, r, sigma),
                black_scholes_rho(S, K, T, r, sigma),
            ],
        })

    # NN Greeks via autograd
    elif model == "Neural Network (BS)":
        nn_model = load_bs_nn()
        inputs = torch.tensor([[S, K, T, r, sigma]], dtype=torch.float32)

        delta = get_delta(nn_model, inputs).item()
        gamma = get_gamma(nn_model, inputs).item()
        vega = get_vega(nn_model, inputs).item()

        st.table({
            "Greek": ["Delta", "Gamma", "Vega"],
            "Value": [delta, gamma, vega]
        })

        st.caption("Greeks computed using PyTorch autograd")

    else:
        st.info("Greeks not available for Monte Carlo.")

# --------------------------------------------------
# TAB 3 — Benchmark
# --------------------------------------------------
with tab_benchmark:
    st.subheader("Benchmark")

    if model == "Neural Network (BS)":
        st.markdown(
            """
            **Neural Network vs Monte Carlo**
            - ~100× faster inference
            - <1% pricing error (see benchmarks in GitHub)
            """
        )
    else:
        st.info("Select Neural Network model to view benchmark summary.")
