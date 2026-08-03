"""Tests für den Experience-Replay-Puffer (agent.ReplayMemory)."""

import numpy as np

from mario.agent import ReplayMemory


def _state():
    return np.zeros((84, 84, 4), dtype=np.uint8)


def test_starts_empty():
    assert len(ReplayMemory(10)) == 0


def test_push_increases_length():
    mem = ReplayMemory(10)
    mem.push(_state(), 1, 1.0, _state(), False)
    assert len(mem) == 1


def test_respects_capacity():
    mem = ReplayMemory(3)
    for i in range(5):
        mem.push(_state(), i, 0.0, _state(), False)
    assert len(mem) == 3


def test_sample_shapes_and_dtypes():
    mem = ReplayMemory(10)
    for i in range(5):
        mem.push(_state(), i, float(i), _state(), i % 2 == 0)
    states, actions, rewards, next_states, dones = mem.sample(4)
    assert states.shape == (4, 84, 84, 4)
    assert next_states.shape == (4, 84, 84, 4)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert dones.shape == (4,)
    assert rewards.dtype == np.float32
    assert dones.dtype == np.float32
