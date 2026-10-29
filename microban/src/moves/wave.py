# src/moves/wave.py
import math


class WaveMove:
    """Wave the right arm."""

    def start(self, obs):
        self.t0 = obs["time"]

    def step(self, obs):
        t = obs["time"] - self.t0
        return {"right_shoulder_roll": -0.8 + 0.4 * math.sin(3 * t)}