"""Gymnasium-Brücke für PPO (Stable-Baselines3) – Phase E.

Nutzt **exakt dieselbe** Bildvorverarbeitung wie das DQN
(`wrappers.create_env`: SkipFrame(4) + Graustufen 84×84 + 4er-FrameStack) und
hüllt das alte gym-Env per **shimmy** in eine Gymnasium-Schnittstelle, die SB3
versteht. Gleiche Beobachtungen wie beim DQN → fairer Algorithmus-Vergleich.

Nur im separaten `.venv-ppo` lauffähig (siehe requirements-ppo.txt).
"""

from __future__ import annotations

from typing import Callable


def make_mario_env(world: int = 1, stage: int = 1):
    """Baut EINE Mario-Umgebung als Gymnasium-Env (für SB3).

    Reihenfolge: unser alter gym-Wrapper-Stack (4er-Tupel-API) → shimmy-Adapter
    → Gymnasium (5er-Tupel: obs, reward, terminated, truncated, info).
    """
    from shimmy import GymV21CompatibilityV0

    from mario.wrappers import create_env

    old_gym_env = create_env(world=world, stage=stage, render=False)
    # shimmy bridget die alte gym-0.21–0.25-API auf Gymnasium – kein Env-Port.
    return GymV21CompatibilityV0(env=old_gym_env)


def mario_env_thunk(world: int = 1, stage: int = 1) -> Callable:
    """Gibt eine parameterlose Factory zurück (für SB3-VecEnvs).

    SB3 erwartet je Parallel-Env eine Funktion, die beim Aufruf eine frische
    Umgebung erzeugt.
    """

    def _init():
        return make_mario_env(world, stage)

    return _init
