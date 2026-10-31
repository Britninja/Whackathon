"""MoveClimb : enchaîne pied droit, pied gauche, main droite, main gauche (touche c)."""
import numpy as np

from moves.movePlaco import BasePlacoMove, MoveRightLeg, MoveLeftLeg, MoveRightArm, MoveLeftArm

ORDER = (MoveRightLeg, MoveLeftLeg, MoveRightArm, MoveLeftArm)


class MoveClimb:
    options = {
        "dz_cm": 7.5,
        "duration_s": 3.0,
        "cycles": 13,
        "settle_min_s": 0.5,   # attente minimale après chaque mouvement
        "settle_max_s": 3.0,   # attente maximale (on repart même si ça bouge encore)
        "settle_speed": 0.2,   # vitesse articulaire max (rad/s) pour dire "c'est stable"
    }
    option_docs = {
        "dz_cm": "montée par mouvement (cm)",
        "duration_s": "durée d'un mouvement (s)",
        "cycles": "nombre de cycles (4 membres par cycle)",
        "settle_min_s": "attente minimale après un mouvement (s)",
        "settle_max_s": "attente maximale après un mouvement (s)",
        "settle_speed": "vitesse articulaire max pour être stable (rad/s)",
    }
    variants = ("climber",)

    CALM_STEPS = 10  # nombre de pas calmes d'affilée (10 pas = 0,2 s à 50 Hz)

    def __init__(self):
        self.moves = [cls() for cls in ORDER]

    def start(self, obs):
        BasePlacoMove.initialized = False  # les cibles repartent de la pose actuelle
        self.count = 0
        self.current = None
        self.settle_start = None
        self.prev_q = np.array(obs["q"], dtype=float)
        self.prev_t = obs["time"]

    def _is_settled(self, obs):
        """Vrai quand le robot a fini de bouger (ou que l'attente maximale est dépassée)."""
        o = self.options
        q = np.array(obs["q"], dtype=float)
        dt = max(obs["time"] - self.prev_t, 1e-3)
        speed = np.max(np.abs(q - self.prev_q)) / dt   # articulation la plus rapide (rad/s)
        self.calm = self.calm + 1 if speed < o["settle_speed"] else 0

        waited = obs["time"] - self.settle_start
        if waited >= o["settle_min_s"] and self.calm >= self.CALM_STEPS:
            return True
        if waited >= o["settle_max_s"]:
            print(f"[MoveClimb] pas stable après {waited:.1f} s (vitesse {speed:.2f} rad/s), on continue")
            return True
        return False

    def step(self, obs):
        out = None
        if self.current is not None:
            out = self.current.step(obs)   # une fois fini, le membre continue de viser sa prise

            if not self.current.done:
                pass
            else:
                # Stabilisation : on attend que le robot ait fini de bouger
                if self.settle_start is None:
                    self.settle_start = obs["time"]
                    self.calm = 0
                settled = self._is_settled(obs)
                if not settled:
                    out = out
                else:
                    self.current = None
                    self.settle_start = None

        self.prev_q = np.array(obs["q"], dtype=float)
        self.prev_t = obs["time"]

        if self.current is not None:
            return out

        # Membre suivant (ou fin de l'escalade)
        if self.count >= 4 * self.options["cycles"]:
            return  # escalade terminée
        self.current = self.moves[self.count % 4]
        self.current.options.update(dz_cm=self.options["dz_cm"], duration_s=self.options["duration_s"])
        self.current.start(obs)
        self.count += 1
        return self.current.step(obs)