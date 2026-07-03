"""Test für die Umgebungsvariablen-Overrides in config."""

import importlib

import config as config_module


def test_env_overrides_apply(monkeypatch):
    monkeypatch.setenv("LEARNING_RATE", "0.001")
    monkeypatch.setenv("EPSILON_DECAY", "500000")
    monkeypatch.setenv("REWARD_CLIP", "1.0")
    try:
        importlib.reload(config_module)
        assert config_module.LEARNING_RATE == 0.001
        assert config_module.EPSILON_DECAY == 500_000
        assert config_module.REWARD_CLIP == 1.0
    finally:
        # Defaults wiederherstellen, damit andere Tests sauberes config sehen
        monkeypatch.delenv("LEARNING_RATE", raising=False)
        monkeypatch.delenv("EPSILON_DECAY", raising=False)
        monkeypatch.delenv("REWARD_CLIP", raising=False)
        importlib.reload(config_module)


def test_defaults_without_env():
    importlib.reload(config_module)
    assert config_module.REWARD_SCALE == 1.0
    assert config_module.REWARD_CLIP is None
    assert config_module.CHECKPOINT_DIR == "checkpoints"
    assert config_module.HIGHLIGHT_DIR == "highlights"


def test_output_dir_overrides(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DIR", "checkpoints_sweep_a")
    monkeypatch.setenv("HIGHLIGHT_DIR", "highlights_sweep_a")
    try:
        importlib.reload(config_module)
        assert config_module.CHECKPOINT_DIR == "checkpoints_sweep_a"
        assert config_module.HIGHLIGHT_DIR == "highlights_sweep_a"
    finally:
        monkeypatch.delenv("CHECKPOINT_DIR", raising=False)
        monkeypatch.delenv("HIGHLIGHT_DIR", raising=False)
        importlib.reload(config_module)
