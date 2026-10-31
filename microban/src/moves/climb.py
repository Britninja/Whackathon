"""MoveClimb : enchaîne pied droit, pied gauche, main droite, main gauche (touche c)."""
from moves.movePlaco import BasePlacoMove, MoveRightLeg, MoveLeftLeg, MoveRightArm, MoveLeftArm

ORDER = (MoveRightLeg, MoveLeftLeg, MoveRightArm, MoveLeftArm)


class MoveClimb:
    options = {"duration_s": 2.0, "cycles": 13}
    option_docs = {
        "duration_s": "durée d'un mouvement de membre (s)",
        "cycles": "nombre de cycles (4 membres par cycle)",
    }
    variants = ("climber",)

    def __init__(self):
        self.options = dict(self.options)
        self.moves = [cls() for cls in ORDER]

    def _next_limb(self):
        """Passe au membre suivant et prépare sa trajectoire."""
        self.current = self.moves[self.limb_index]
        self.current.options["duration_s"] = self.options["duration_s"]
        self.current.begin()
        self.limb_index = (self.limb_index + 1) % len(self.moves)
        self.count += 1

    def start(self, obs):
        BasePlacoMove.setup(obs)  # repart de la pose de départ à chaque appui sur c
        self.limb_index = 0
        self.count = 0
        self.movement_time = 0.0
        self._next_limb()

    def step(self, obs):
        dt = BasePlacoMove.time_step(obs)
        self.movement_time += dt

        # Mouvement en cours, ou fini : on passe au membre suivant dans le même pas
        if self.movement_time <= self.options["duration_s"]:
            self.current.set_target(self.movement_time)
        else:
            self._next_limb()
            self.movement_time = 0.0
            self.current.set_target(0.0)

        targets = BasePlacoMove.solve(dt)
        if self.count >= 4 * self.options["cycles"] + 1:
            return  # escalade terminée
        return targets