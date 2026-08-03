"""Tests für den Agenten (agent.MarioAgent) auf CPU."""

import numpy as np
import torch

from mario import config
from mario.agent import MarioAgent
from mario.model import MarioNet


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


def test_load_weights_transfers_net_but_keeps_exploration(tmp_path):
    """Warm-Start: Netzgewichte übernehmen, aber Epsilon/Step-Count frisch lassen."""
    source = _agent()
    source.step_count = 12345
    source.epsilon = 0.02
    ckpt = tmp_path / "src.pt"
    source.save(path=str(tmp_path), name="src.pt")

    target = _agent()  # frischer Agent: Epsilon=EPSILON_START, step_count=0
    fresh_epsilon, fresh_steps = target.epsilon, target.step_count
    target.load_weights(str(ckpt))

    # Gewichte wurden übernommen ...
    for p_src, p_tgt in zip(
        source.online_net.parameters(), target.online_net.parameters()
    ):
        assert torch.equal(p_src, p_tgt)
    # ... aber Exploration/Step-Count blieben frisch (nicht 0.02 / 12345).
    assert target.epsilon == fresh_epsilon
    assert target.step_count == fresh_steps


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
