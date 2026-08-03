"""Tests für die Episoden-Aufnahme (record.record_frames) mit Fake-Objekten."""

import numpy as np

from mario.record import record_frames


class FakeAgent:
    def act(self, state):
        return 0


class FakeEnv:
    """Beendet nach ``n`` Schritten; render() liefert ein Dummy-RGB-Bild."""

    def __init__(self, n):
        self.n = n
        self.i = 0

    def reset(self):
        self.i = 0
        return 0

    def render(self, mode="rgb_array"):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def step(self, action):
        self.i += 1
        done = self.i >= self.n
        return 0, 1.0, done, {"x_pos": self.i * 10, "flag_get": done}


def test_records_one_frame_per_step():
    frames, info = record_frames(FakeAgent(), FakeEnv(3))
    assert len(frames) == 3
    assert info["x_pos"] == 30
    assert info["flag_get"] is True


def test_respects_max_steps():
    frames, _ = record_frames(FakeAgent(), FakeEnv(1000), max_steps=5)
    assert len(frames) == 5
