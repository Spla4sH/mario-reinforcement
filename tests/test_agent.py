"""Tests für den Agenten (agent.MarioAgent) auf CPU."""

import numpy as np
import torch

import config
from agent import MarioAgent
from model import MarioNet


def _state():
    return np.zeros((84, 84, 4), dtype=np.uint8)


def _agent():
    return MarioAgent(7, device=torch.device("cpu"))


def test_act_returns_valid_action():
    action = _agent().act(_state())
    assert 0 <= action < 7


def test_select_action_decays_epsilon_and_counts_steps():
    agent = _agent()
    start_epsilon = agent.epsilon
    for _ in range(100):
        agent.select_action(_state())
    assert agent.step_count == 100
    assert agent.epsilon < start_epsilon


def test_learn_returns_none_when_memory_too_small():
    assert _agent().learn() is None


def test_forward_handles_non_contiguous_input():
    """Regression: das Netz muss auch nicht-zusammenhängende Tensoren verkraften.

    Im Training werden States via ``permute(0, 3, 1, 2)`` von (B, H, W, C) nach
    (B, C, H, W) umsortiert – das ergibt einen nicht-contiguous Tensor. Früher
    crashte ``x.view(...)`` in ``MarioNet.forward`` genau hier
    ("view size is not compatible ...").
    """
    net = MarioNet(config.FRAME_STACK, 7)
    x = torch.zeros(8, 84, 84, config.FRAME_STACK, dtype=torch.uint8).permute(
        0, 3, 1, 2
    )
    assert not x.is_contiguous()
    out = net(x)
    assert out.shape == (8, 7)


def test_learn_runs_full_step_and_returns_loss():
    """Regression: ein echter Lernschritt (voller Batch) muss durchlaufen.

    Deckt den Pfad ab, der beim Smoke Test crashte – vorher prüften die Tests
    nur den leeren Replay-Buffer.
    """
    agent = _agent()
    for _ in range(config.MIN_MEMORY):
        agent.memory.push(_state(), 0, 1.0, _state(), False)
    loss = agent.learn()
    assert isinstance(loss, float)
    assert np.isfinite(loss)
