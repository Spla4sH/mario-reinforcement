"""Reward-Shaping – bewusst ohne schwere Abhängigkeiten (gut testbar).

Die eigentliche Gym-Integration (``RewardWrapper``) liegt in ``wrappers.py`` und
nutzt diese reine Funktion. So lässt sich die Logik isoliert testen.
"""

from __future__ import annotations


def shape_reward(reward: float, scale: float = 1.0, clip: float | None = None) -> float:
    """Skaliert und/oder beschneidet eine Belohnung.

    Args:
        reward: Roh-Belohnung der Umgebung.
        scale: Teiler zur Skalierung (z. B. 10.0). ``1.0`` lässt sie unverändert.
        clip: Falls gesetzt, wird auf ``[-clip, clip]`` beschnitten.

    Returns:
        Die transformierte Belohnung als float.
    """
    shaped = float(reward) / scale
    if clip is not None:
        shaped = max(-clip, min(clip, shaped))
    return shaped
