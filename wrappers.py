"""Environment-Wrapper für Bildvorverarbeitung."""

from collections import deque

import cv2
import gym
import numpy as np
from gym.spaces import Box

import config
from reward import shape_reward


class SkipFrame(gym.Wrapper):
    """Wiederholt eine Aktion für mehrere Frames und summiert die Belohnung."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class RewardWrapper(gym.RewardWrapper):
    """Skaliert und/oder beschneidet die Belohnung (siehe reward.shape_reward)."""

    def __init__(self, env, scale=1.0, clip=None):
        super().__init__(env)
        self._scale = scale
        self._clip = clip

    def reward(self, reward):
        return shape_reward(reward, self._scale, self._clip)


class ProgressReward(gym.Wrapper):
    """Ersetzt den Δx-Bewegungs-Reward durch einen echten Fortschritts-Reward.

    Belohnt nur einen *neuen*, plausiblen Vorwärtsschritt (x wächst über das
    bisherige Maximum, aber um höchstens ``max_step`` Pixel/Frame). Damit bringt
    Im-Kreis-Laufen in Loop-/Labyrinth-Levels nichts mehr (kein neues Maximum),
    die Zeitstrafe wirkt sogar dagegen.

    Der ``max_step``-Deckel fängt zwei unphysikalische Sprünge ab, die der Agent
    sonst als Reward-Exploit missbraucht: den 16-Bit-Overflow der x-Position
    (``x_pos`` springt in 4-4 kurz auf 65535) und Screen-/Loop-Teleports. Ohne
    ihn farmte ein Agent den 65535-Glitch für ~65000 Reward auf einen Schlag.

    Innerster Wrapper (direkt auf dem NES-Env, vor SkipFrame): sieht jeden Frame
    und die rohe x-Position. Gegen das Reward-Farming in 4-4 (per PROGRESS_REWARD).
    """

    def __init__(self, env, time_penalty=0.1, flag_bonus=50.0, max_step=64):
        super().__init__(env)
        self._time_penalty = time_penalty
        self._flag_bonus = flag_bonus
        self._max_step = max_step
        self._max_x = 0

    def reset(self, **kwargs):
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        x = int(info.get("x_pos", 0))
        delta = x - self._max_x
        # Nur plausible Vorwärtsschritte zählen; Glitches/Teleports (delta riesig)
        # und Rückschritte (delta <= 0) geben keinen Fortschritts-Reward.
        if 0 < delta <= self._max_step:
            reward = float(delta)
            self._max_x = x
        else:
            reward = 0.0
        reward -= self._time_penalty
        if info.get("flag_get"):
            reward += self._flag_bonus
        return obs, float(reward), done, info


class GrayScaleResize(gym.ObservationWrapper):
    """Konvertiert zu Graustufen und skaliert auf 84x84 für das Netz."""

    def __init__(self, env, size=84):
        super().__init__(env)
        self._size = size
        self.observation_space = Box(
            low=0, high=255, shape=(size, size, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self._size, self._size), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]


class FrameStack(gym.Wrapper):
    """Stapelt die letzten N Frames als Eingabe für das Netz."""

    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self._num_stack = num_stack
        self._frames = deque(maxlen=num_stack)
        low = np.repeat(env.observation_space.low, num_stack, axis=-1)
        high = np.repeat(env.observation_space.high, num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=np.uint8)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.concatenate(list(self._frames), axis=-1)


def create_env(world=1, stage=1, render=True, action_set=None):
    """Erstellt die Super Mario Umgebung mit allen Wrappern.

    Das Original-Spielbild wird im Fenster angezeigt (render=True),
    während das Netz intern verkleinerte Graustufen-Frames bekommt.

    ``action_set`` überschreibt ``config.ACTION_SET`` für diesen einen Aufruf.
    Nötig, sobald mehrere Modelle in einem Prozess laufen (Demo-App, gen_demos):
    das 8-4-Modell braucht 8 Aktionen, alle anderen 7 – eine Umgebungsvariable
    kann das nicht pro Level unterscheiden.
    """
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    # Suffix -v0 = Originalgrafik des NES-Spiels (ROM ist im Paket enthalten).
    env_id = f"SuperMarioBros-{world}-{stage}-v0"
    env = gym_super_mario_bros.make(env_id)

    # Vereinfachte Aktionen: SIMPLE_MOVEMENT hat 7 Aktionen
    # [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]
    #
    # ACTION_SET="down" ergaenzt eine 8. Aktion ['down']: In den Labyrinth-Schloessern
    # (v. a. 8-4) fuehrt der richtige Weg durch Roehren, und ohne "down" kann Mario
    # nicht einsteigen – das Level waere prinzipiell unloesbar. Default bleibt bei
    # 7 Aktionen, damit alle bisherigen Modelle unveraendert weiterlaufen.
    gewaehlt = (action_set or config.ACTION_SET).strip().lower()
    actions = SIMPLE_MOVEMENT + [["down"]] if gewaehlt == "down" else SIMPLE_MOVEMENT
    env = JoypadSpace(env, actions)

    # Fortschritts-Reward (opt-in) direkt auf dem rohen NES-Env: sieht die echte
    # x-Position pro Frame, ersetzt den Δx-Reward gegen Loop-Farming (z. B. 4-4).
    if config.PROGRESS_REWARD:
        env = ProgressReward(env)

    # Wrapper anwenden
    env = SkipFrame(env, skip=config.FRAME_SKIP)
    # Reward-Shaping (Default scale=1.0/clip=None => unverändert; per config/Env tunebar)
    env = RewardWrapper(env, scale=config.REWARD_SCALE, clip=config.REWARD_CLIP)
    env = GrayScaleResize(env, size=config.FRAME_SIZE)
    env = FrameStack(env, num_stack=config.FRAME_STACK)

    return env
