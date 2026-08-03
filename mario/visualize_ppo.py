"""KI-Vision für PPO: Grad-CAM auf SB3s CnnPolicy – was sieht der PPO-Agent?

Gleiche Technik wie `visualize.py` (DQN), aber auf dem Actor-Kopf der PPO-Policy:
Die Heatmap zeigt, welche Bildregionen das **Logit der gewählten Aktion** treiben.
Wiederverwendet werden `GradCAM` und `make_overlay` aus visualize.py – nur das
„Netz" ist hier ein dünner Wrapper, der die Aktions-Logits der Policy liefert.

**Nur im `.venv-ppo` lauffähig** (braucht Stable-Baselines3):
    .venv-ppo\\Scripts\\python visualize_ppo.py --model checkpoints_ppo/mario_ppo_gen_welt1.zip \\
        --world 1 --stage 1 --episodes 1 --save vision_ppo.gif --no-window
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mario.visualize import GradCAM, make_overlay, _save_gif


class PolicyLogits(torch.nn.Module):
    """Macht die PPO-Policy für GradCAM „netzförmig": Eingabe-Tensor -> Aktions-Logits."""

    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.policy.get_distribution(obs_t).distribution.logits


def visualize_ppo(
    model_path: str,
    world: int,
    stage: int,
    episodes: int = 1,
    save: str | None = None,
    scale: int = 2,
    show_window: bool = True,
) -> None:
    if not Path(model_path).exists():
        print(f"Kein PPO-Modell gefunden unter: {model_path}")
        return

    import cv2  # lazy: nur für Anzeige/Overlay nötig
    from stable_baselines3 import PPO

    from mario.wrappers import create_env

    model = PPO.load(model_path)
    policy = model.policy
    policy.set_training_mode(False)

    # Letzter Conv-Layer von SB3s NatureCNN als Grad-CAM-Ziel (wie features[4] beim DQN).
    target_layer = policy.features_extractor.cnn[4]
    cam = GradCAM(PolicyLogits(policy), target_layer)

    env = create_env(world=world, stage=stage, render=False)
    frames: list[np.ndarray] = []
    if show_window:
        print("Fenster mit PPO-Vision-Overlay öffnet sich. Beenden mit 'q'.")

    for episode in range(1, episodes + 1):
        state = env.reset()
        info: dict = {}
        while True:
            obs_t, _ = policy.obs_to_tensor(state)
            action, heat = cam(obs_t)
            # PPO-Logit-Gradienten haben einen höheren Grundpegel als DQN-Q-Werte
            # -> Floor abziehen und quadrieren, sonst liegt ein flächiger Schleier
            # über dem Bild statt klarer Hotspots.
            heat = ((heat - heat.min()) / (heat.max() - heat.min() + 1e-8)) ** 2

            rgb = env.render(mode="rgb_array")
            overlay = make_overlay(rgb, heat, scale=scale)
            if save:
                frames.append(overlay)

            if show_window:
                cv2.imshow("Mario - PPO-Vision (Grad-CAM)", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    env.close()
                    cv2.destroyAllWindows()
                    _save_gif(frames, save)
                    return

            state, _, done, info = env.step(action)
            if done:
                break
        print(f"Episode {episode}: x={info.get('x_pos', 0)} | Flagge: {bool(info.get('flag_get'))}")

    env.close()
    if show_window:
        cv2.destroyAllWindows()
    _save_gif(frames, save)
    print("Fertig.")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO-KI-Vision-Overlay (Grad-CAM).")
    parser.add_argument("--model", default="checkpoints_ppo/mario_ppo_gen_welt1.zip")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--save", type=str, default=None, help="Pfad für ein GIF")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--no-window", action="store_true", help="headless, nur GIF-Export")
    args = parser.parse_args()
    visualize_ppo(
        args.model, args.world, args.stage, args.episodes,
        args.save, args.scale, show_window=not args.no_window,
    )


if __name__ == "__main__":
    main()
