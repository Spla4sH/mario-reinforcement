"""Greedy-Evaluation des Agenten – der ehrliche Fortschritts-Indikator.

Anders als der Trainings-Reward (verrauscht durch Epsilon-Exploration) misst die
Evaluation rein greedy: Wie oft erreicht Mario die Flagge, wie weit kommt er im
Schnitt? Bewusst ohne schwere Importe gehalten (nur die übergebenen Objekte),
damit die Logik isoliert testbar ist.
"""

from __future__ import annotations


def evaluate(agent, env, episodes: int = 5, max_steps: int = 100_000) -> dict:
    """Spielt ``episodes`` Episoden greedy und gibt aggregierte Metriken zurück.

    Args:
        agent: Objekt mit ``act(state) -> action`` (greedy, ohne Seiteneffekte).
        env: Gym-artige Umgebung mit ``reset()`` und ``step(action)``.
        episodes: Anzahl der Evaluationsepisoden.
        max_steps: Sicherheitskappe gegen Endlos-Episoden.

    Returns:
        Dict mit ``flag_rate`` (Anteil mit Flagge), ``mean_x`` (mittlere
        x-Position), ``max_x`` und ``mean_reward``.
    """
    episodes = max(1, episodes)
    flags = 0
    x_positions: list[int] = []
    rewards: list[float] = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        info: dict = {}
        for _ in range(max_steps):
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        flags += int(bool(info.get("flag_get", False)))
        x_positions.append(int(info.get("x_pos", 0)))
        rewards.append(total_reward)

    return {
        "flag_rate": flags / episodes,
        "mean_x": sum(x_positions) / len(x_positions),
        "max_x": max(x_positions),
        "mean_reward": sum(rewards) / len(rewards),
    }
