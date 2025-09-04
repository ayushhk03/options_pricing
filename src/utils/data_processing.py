import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_and_preprocess_data(model_type="black_scholes"):
    """
    Load and preprocess data for training

    Parameters:
    model_type: "black_scholes" or "heston"

    Returns:
    Processed data loaders and scalers
    """
    if model_type == "black_scholes":
        X = np.load('../data/processed/X_black_scholes.npy')
        y = np.load('../data/processed/y_black_scholes.npy')
    else:  # heston
        X = np.load('../data/processed/X_heston.npy')
        y = np.load('../data/processed/y_heston.npy')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Scale features
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)

    # Scale targets (prices can vary widely)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'X_test_raw': X_test,
        'y_test_raw': y_test
    }


def prepare_inference_data(input_params, X_scaler, model_type="black_scholes"):
    """
    Prepare data for inference

    Parameters:
    input_params: Dictionary of input parameters
    X_scaler: Fitted scaler for features

    Returns:
    Prepared tensor for model inference
    """
    if model_type == "black_scholes":
        # Expected order: [S, K, T, r, sigma]
        features = np.array([
            input_params['S'],
            input_params['K'],
            input_params['T'],
            input_params['r'],
            input_params['sigma']
        ])
    else:  # heston
        # Expected order: [S, K, T, r, v0, kappa, theta, sigma, rho]
        features = np.array([
            input_params['S'],
            input_params['K'],
            input_params['T'],
            input_params['r'],
            input_params['v0'],
            input_params['kappa'],
            input_params['theta'],
            input_params['sigma'],
            input_params['rho']
        ])

    # Scale features
    features_scaled = X_scaler.transform(features.reshape(1, -1))

    return torch.FloatTensor(features_scaled)


def inverse_transform_prices(scaled_prices, y_scaler):
    """
    Inverse transform scaled prices back to original scale
    """
    return y_scaler.inverse_transform(scaled_prices.reshape(-1, 1)).flatten()