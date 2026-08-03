"""Test für die Grad-CAM-Kernlogik (visualize.GradCAM) – ohne Spiel/OpenCV."""

import torch

from mario.model import MarioNet
from mario.visualize import GradCAM


def test_gradcam_returns_valid_action_and_heatmap():
    net = MarioNet(4, 7).eval()
    cam = GradCAM(net, net.features[4])

    state = torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.uint8)
    action, heat = cam(state)

    assert 0 <= action < 7
    assert heat.shape == (7, 7)  # letzter Conv-Layer bei 84x84-Eingabe
    assert float(heat.min()) >= 0.0
    assert float(heat.max()) <= 1.0 + 1e-6
