"""Tests für den Agenten (agent.MarioAgent) auf CPU."""

import numpy as np
import torch

from agent import MarioAgent


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
