"""KI-Vision-Overlay: zeigt per Grad-CAM, worauf das CNN beim Spielen achtet.

Legt eine Heatmap (rot = wichtig für die gewählte Aktion) über das Original-
Spielbild. Optional als GIF speichern – ideal fürs README.

Aufruf:
    python visualize.py [--episodes N] [--checkpoint pfad] [--save out.gif] [--scale 2]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config
from wrappers import create_env
from agent import MarioAgent


class GradCAM:
    """Grad-CAM für den letzten Conv-Layer des Online-Netzes.

    Liefert pro Frame die gewählte (greedy) Aktion und eine normalisierte
    Heatmap, die zeigt, welche Bildregionen den Q-Wert dieser Aktion treiben.
    """

    def __init__(self, net, target_layer):
        self.net = net
        self._activations = None
        self._gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0]

    def __call__(self, state_t):
        """state_t: (1, C, H, W) uint8. Gibt (action, cam[h,w] float32) zurück."""
        q_values = self.net(state_t)
        action = int(q_values.argmax(dim=1).item())

        self.net.zero_grad()
        q_values[0, action].backward()

        # Grad-CAM: Kanäle nach mittlerem Gradienten gewichten
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1)).squeeze(0)
        cam = cam / (cam.max() + 1e-8)
        return action, cam.detach().cpu().numpy()


def make_overlay(rgb_frame, cam, scale=2, alpha=0.5):
    """Legt die Grad-CAM-Heatmap über das RGB-Originalbild (Rückgabe in RGB)."""
    h, w = rgb_frame.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_uint8 = np.uint8(np.clip(cam_resized, 0, 1) * 255)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(rgb_frame, 1 - alpha, heatmap_rgb, alpha, 0)
    if scale != 1:
        overlay = cv2.resize(
            overlay, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
        )
    return overlay


def visualize(episodes=2, checkpoint="checkpoints/mario_agent.pt", save=None, scale=2):
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        print(f"Kein Checkpoint gefunden unter: {checkpoint_path}")
        print("Trainiere zuerst mit 'python train.py'.")
        return

    env = create_env(world=config.WORLD, stage=config.STAGE, render=False)
    agent = MarioAgent(env.action_space.n)
    agent.load(str(checkpoint_path))
    agent.online_net.eval()

    # Letzter Conv-Layer (features[4]) als Grad-CAM-Ziel
    cam = GradCAM(agent.online_net, agent.online_net.features[4])

    frames = []  # für optionales GIF
    print("Fenster mit Vision-Overlay öffnet sich. Beenden mit 'q'.")

    for episode in range(1, episodes + 1):
        state = env.reset()
        info = {}
        while True:
            state_t = (
                torch.tensor(state, dtype=torch.uint8)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(agent.device)
            )
            action, heat = cam(state_t)

            rgb = env.render(mode="rgb_array")  # Original-NES-Bild (RGB)
            overlay = make_overlay(rgb, heat, scale=scale)
            if save:
                frames.append(overlay)

            cv2.imshow(
                "Mario - KI-Vision (Grad-CAM)",
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                env.close()
                cv2.destroyAllWindows()
                _save_gif(frames, save)
                return

            state, _, done, info = env.step(action)
            if done:
                break

        print(f"Episode {episode}: Position {info.get('x_pos', 0)}")

    env.close()
    cv2.destroyAllWindows()
    _save_gif(frames, save)
    print("Fertig.")


def _save_gif(frames, save):
    if not save or not frames:
        return
    try:
        import imageio
    except ImportError:
        print("Zum GIF-Speichern bitte 'pip install imageio' (siehe requirements.txt).")
        return
    imageio.mimsave(save, frames, fps=30)
    print(f"GIF gespeichert: {save} ({len(frames)} Frames)")


def main():
    parser = argparse.ArgumentParser(description="KI-Vision-Overlay (Grad-CAM).")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mario_agent.pt")
    parser.add_argument("--save", type=str, default=None, help="Pfad für ein GIF, z. B. vision.gif")
    parser.add_argument("--scale", type=int, default=2, help="Vergrößerungsfaktor der Anzeige")
    args = parser.parse_args()
    visualize(args.episodes, args.checkpoint, args.save, args.scale)


if __name__ == "__main__":
    main()
