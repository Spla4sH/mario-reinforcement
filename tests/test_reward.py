"""Tests für das Reward-Shaping (reward.shape_reward)."""

from mario.reward import shape_reward


def test_scale_divides():
    assert shape_reward(10.0, scale=10.0) == 1.0


def test_default_is_noop():
    assert shape_reward(5.0) == 5.0


def test_clip_upper_and_lower():
    assert shape_reward(100.0, clip=1.0) == 1.0
    assert shape_reward(-100.0, clip=1.0) == -1.0


def test_clip_within_bounds_unchanged():
    assert shape_reward(0.5, clip=1.0) == 0.5


def test_scale_then_clip():
    # 50 / 10 = 5 -> auf 1.0 beschnitten
    assert shape_reward(50.0, scale=10.0, clip=1.0) == 1.0


def test_returns_float():
    assert isinstance(shape_reward(3), float)
