import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """
    Train the neural network model
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses


def generate_dataset(model_type, n_samples=100000):
    """
    Generate training dataset for the specified model
    """
    # This would generate appropriate training data based on the model type
    # For simplicity, we'll return dummy data
    if model_type == "black_scholes":
        # Inputs: S, K, T, r, sigma
        inputs = torch.randn(n_samples, 5)
        # Generate prices using Black-Scholes formula (simplified)
        targets = torch.randn(n_samples, 1)
    else:  # heston
        # Inputs: S, K, T, r, v0, kappa, theta, sigma, rho
        inputs = torch.randn(n_samples, 9)
        # Generate prices using Heston model (simplified)
        targets = torch.randn(n_samples, 1)

    return inputs, targets