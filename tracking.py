"""Optionales Experiment-Tracking mit Weights & Biases (W&B).

Bewusst ausfallsicher: Ist Tracking deaktiviert, ``wandb`` nicht installiert oder
der Login fehlt, wird der Tracker zur No-Op – das Training läuft unverändert weiter.
Aktivierung über ``USE_WANDB=1`` (siehe config.py).
"""

from __future__ import annotations


class Tracker:
    """Dünne, ausfallsichere Hülle um ``wandb``."""

    def __init__(
        self,
        enabled: bool,
        config_dict: dict | None = None,
        project: str = "mario-rl",
        run_name: str | None = None,
    ) -> None:
        self.run = None
        self._wandb = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError:
            print("[Tracking] wandb nicht installiert – überspringe (pip install wandb).")
            return
        try:
            self.run = wandb.init(project=project, name=run_name, config=config_dict)
            self._wandb = wandb
            print(f"[Tracking] W&B aktiv: {self.run.url}")
        except Exception as exc:  # Login/Netzwerk/Konfiguration
            print(f"[Tracking] W&B-Init fehlgeschlagen ({exc}); fahre ohne Tracking fort.")
            self.run = None

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.run is None:
            return
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self._wandb.finish()
            self.run = None
