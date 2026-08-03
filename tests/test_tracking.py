"""Tests für die ausfallsichere W&B-Hülle (tracking.Tracker).

In der Testumgebung ist ``wandb`` nicht installiert – der Tracker muss in jedem
Fall geräuschlos zur No-Op werden, ohne das Training zu stören.
"""

from mario.tracking import Tracker


def test_disabled_is_noop():
    t = Tracker(enabled=False)
    assert t.run is None
    # darf nicht werfen
    t.log({"reward": 1.0}, step=1)
    t.finish()


def test_enabled_without_wandb_is_graceful():
    # wandb ist im Test-/CI-Env nicht installiert -> ImportError wird abgefangen
    t = Tracker(enabled=True, config_dict={"lr": 0.001}, run_name="x")
    assert t.run is None
    t.log({"reward": 1.0}, step=1)
    t.finish()
