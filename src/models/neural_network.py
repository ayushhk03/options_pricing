import torch
import torch.nn as nn


class OptionPricingNN(nn.Module):
    """
    Neural network for option pricing
    """

    def __init__(self, input_dim, hidden_layers=[64, 64, 32]):
        super(OptionPricingNN, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class HestonNN(nn.Module):
    """
    Specialized neural network for Heston model
    """

    def __init__(self, input_dim, hidden_layers=[128, 128, 64, 32]):
        super(HestonNN, self).__init__()

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Price prediction head
        self.price_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        price = self.price_head(features)
        return price