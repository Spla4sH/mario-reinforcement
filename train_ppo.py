"""PPO-Training für Mario via Stable-Baselines3 (Phase E).

Der Algorithmus-Upgrade gegenüber dem DQN: on-policy Policy-Gradient, parallele
Rollouts, stabiler & sample-effizienter. Gleiche Bildvorverarbeitung wie das DQN
(siehe mario_ppo_env), damit der Vergleich fair ist.

**Nur im separaten `.venv-ppo` lauffähig** (nicht im gepinnten DQN-.venv!):
    .venv-ppo\\Scripts\\python train_ppo.py --world 1 --stage 2 --timesteps 2000000

Erstes Ziel: Level 1-2 (für DQN zu hart). Fortschritt live via TensorBoard:
    .venv-ppo\\Scripts\\python -m tensorboard.main --logdir runs_ppo
"""

from __future__ import annotations

import argparse
import os
from typing import Callable


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linearer Decay: LR fällt über den Lauf von initial_value auf ~0.

    Stabilisiert die späte Trainingsphase (weniger Zappeln, wenn die Policy
    schon gut ist). SB3 ruft die Funktion mit ``progress_remaining`` (1→0) auf.
    """

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def main() -> None:
    parser = argparse.ArgumentParser(description="Mario-Agent mit PPO trainieren.")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=2, help="Standard: 1-2 (DQN-Grenze)")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8, help="Parallele Umgebungen")
    parser.add_argument("--logdir", type=str, default="runs_ppo")
    parser.add_argument("--save", type=str, default="checkpoints_ppo/mario_ppo")
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Pfad zu einem PPO-.zip: laden und weitertrainieren (Steps zählen weiter).",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropie-Bonus: höher = länger explorieren (gegen vorzeitige Konvergenz).",
    )
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import (
        SubprocVecEnv,
        VecMonitor,
        VecNormalize,
    )

    from mario_ppo_env import mario_env_thunk

    run_name = f"ppo_{args.world}-{args.stage}"

    # Parallele Umgebungen (je eigener Prozess – robuster für den NES-Emulator).
    env = SubprocVecEnv(
        [mario_env_thunk(args.world, args.stage) for _ in range(args.n_envs)]
    )
    env = VecMonitor(env)  # loggt ROHE Episoden-Rewards (vor Normalisierung)

    # Reward-Normalisierung stabilisiert PPO bei stark schwankenden Returns
    # (gegen die Kollaps-Oszillation). Bilder NICHT normalisieren – nur Rewards.
    vecnorm_path = (
        args.resume_from.replace(".zip", "_vecnormalize.pkl") if args.resume_from else ""
    )
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        print(f"VecNormalize-Statistik geladen: {vecnorm_path}")
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # PPO mit CNN-Policy (SB3s NatureCNN ≈ unser DQN-Netz). Hyperparameter an
    # der Atari-PPO-Praxis orientiert – Startpunkt zum Tunen.
    if args.resume_from and os.path.exists(args.resume_from):
        model = PPO.load(args.resume_from, env=env, tensorboard_log=args.logdir)
        reset_timesteps = False
        print(f"Fortsetzen von {args.resume_from} (bisher {model.num_timesteps:,} Steps)")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=args.logdir,
            n_steps=512,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            learning_rate=linear_schedule(2.5e-4),
        )
        reset_timesteps = True

    print(
        f"PPO-Training startet: World {args.world}-{args.stage} | "
        f"{args.n_envs} Envs | {args.timesteps:,} Steps"
    )

    # Checkpoints regelmäßig sichern (resumebar / für Demo).
    ckpt = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path="checkpoints_ppo",
        name_prefix=run_name,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=ckpt,
        tb_log_name=run_name,
        reset_num_timesteps=reset_timesteps,
    )
    model.save(args.save)
    env.save(f"{args.save}_vecnormalize.pkl")  # Normalisierungs-Statistik mitsichern
    env.close()
    print(f"Fertig. Modell gespeichert: {args.save}.zip (+ _vecnormalize.pkl)")


if __name__ == "__main__":
    main()
