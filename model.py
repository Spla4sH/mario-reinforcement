"""Convolutional Neural Network für den DQN-Agenten."""

import torch
import torch.nn as nn


class MarioNet(nn.Module):
    """CNN das aus gestapelten Frames Q-Werte für jede Aktion berechnet.

    Architektur orientiert sich am Nature DQN Paper (Mnih et al., 2015).
    """

    def __init__(self, input_channels, num_actions):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        # Eingabe: (batch, channels, 84, 84), Werte 0-255 -> normalisieren
        x = x.float() / 255.0
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
