"""Kernmodule des Mario-RL-Projekts.

Hier liegt alles, was von mehreren Stellen importiert wird: Agent und Netz,
die Environment-Wrapper, Reward-Shaping, Konfiguration, Metriken sowie die
Grad-CAM-Visualisierung sowie die Level-Tabelle der Demo (``demo``).

Ausfuehrbare Skripte liegen daneben in ``scripts/``. Damit die das Paket finden,
wird es einmal je venv installiert:

    pip install -e .

Module mit eigener Kommandozeile, die zugleich Bibliothek sind, startet man als
Modul, z. B.::

    python -m mario.visualize --episodes 1 --save assets/vision.gif
"""
