"""Tests für die Greedy-Evaluation (evaluate.evaluate) mit Fake-Objekten."""

from evaluate import evaluate


class FakeAgent:
    def act(self, state):
        return 0


class FakeEnv:
    """Spielt ein vorgegebenes Skript ab: pro Episode eine Liste von (done, info)."""

    def __init__(self, script):
        self.script = script
        self.ep = -1
        self.step_i = 0

    def reset(self):
        self.ep += 1
        self.step_i = 0
        return 0

    def step(self, action):
        done, info = self.script[self.ep][self.step_i]
        self.step_i += 1
        return 0, 1.0, done, info


def test_aggregates_metrics():
    script = [
        [(True, {"flag_get": True, "x_pos": 3000})],
        [(False, {"x_pos": 500}), (True, {"flag_get": False, "x_pos": 1000})],
    ]
    res = evaluate(FakeAgent(), FakeEnv(script), episodes=2)
    assert res["flag_rate"] == 0.5
    assert res["mean_x"] == 2000
    assert res["max_x"] == 3000
    assert res["mean_reward"] == 1.5  # ep0: 1 Schritt, ep1: 2 Schritte


def test_episodes_clamped_to_at_least_one():
    script = [[(True, {"flag_get": False, "x_pos": 10})]]
    res = evaluate(FakeAgent(), FakeEnv(script), episodes=0)
    assert res["flag_rate"] == 0.0
    assert res["max_x"] == 10
