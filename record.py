"""Aufnahme einer greedy Episode als Frame-Liste / GIF.

Wird für das Auto-Highlight-Reel (train.py) und die Gradio-Demo (app.py) genutzt.
``record_frames`` ist bewusst frei von schweren Importen (gut testbar); das
GIF-Speichern (``save_gif``) lädt ``imageio`` erst bei Bedarf.
"""

from __future__ import annotations


def record_frames(agent, env, max_steps: int = 100_000):
    """Spielt eine greedy Episode und sammelt die Original-RGB-Frames.

    Args:
        agent: Objekt mit ``act(state) -> action``.
        env: Umgebung mit ``reset()``, ``step()`` und ``render(mode="rgb_array")``.
        max_steps: Sicherheitskappe gegen Endlos-Episoden.

    Returns:
        ``(frames, info)`` – Liste der RGB-Frames und das letzte info-Dict.
    """
    state = env.reset()
    frames = []
    info: dict = {}
    for _ in range(max_steps):
        frames.append(env.render(mode="rgb_array"))
        state, _, done, info = env.step(agent.act(state))
        if done:
            break
    return frames, info


def save_gif(frames, path: str, fps: int = 30, loop: int = 0) -> bool:
    """Speichert Frames als GIF. Gibt True bei Erfolg zurück.

    ``loop=0`` bedeutet Endlos-Schleife (sonst spielt das GIF nur einmal ab).
    """
    if not frames:
        return False
    import imageio

    imageio.mimsave(path, frames, fps=fps, loop=loop)
    return True
