import numpy as np
import placo
import microban_model
from microban_model import JOINTS

# Ce script sert à bouger la boule du bras droit du robot vers le haut, avec des parametres ajustables
# Pour cela, il utilise un solver placo qui update au fur et a mesure la position cible de cette boule vers la position demandé


HAND_FRAME = "right_hand_ball"                       # la frame que Placo déplace
ARM_JOINTS = ("right_shoulder_pitch", "right_shoulder_roll", "right_elbow")


class movePlacoMove:
    """Bras droit : la main monte de quelques cm (cinématique inverse avec Placo)."""

    variants = ("climber",)                          # pour le robot grimpeur seulement
    options = {"lift_cm": 3.0, "duration_s": 2.0, "solver_iters": 5}
    option_docs = {
        "lift_cm": "de combien la main monte (cm)",
        "duration_s": "durée du mouvement (s)",
        "solver_iters": "pas de solveur par tick (plus = suit mieux la cible)",
    }

    def __init__(self):
        """
        Rôle    : prépare Placo UNE SEULE FOIS, au démarrage du programme.
                  (Charger le modèle est lent : on ne le fait jamais dans step.)
        Attention : on appelle nos objets self.model / self.solver, jamais
                  self.robot (ce nom est réservé par le programme de microban).
        """
        self.model = placo.RobotWrapper(str(microban_model.urdf(climb=True)))
        self.solver = placo.KinematicsSolver(self.model)
        self.solver.mask_fbase(True)                 # tronc fixe : seul le bras bouge
        self.model.update_kinematics()

        # La tâche est créée ici, une fois. Cible provisoire : la main là où elle est.
        self.task = self.solver.add_position_task(HAND_FRAME, self._hand())
        self.task.configure("main", "soft", 1.0)

    def _hand(self):
        """
        Rôle    : position actuelle de la main dans le modèle Placo.
        Retour  : tableau numpy (x, y, z) en mètres, repère du tronc.
        """
        return self.model.get_T_world_frame(HAND_FRAME)[:3, 3].copy()

    def start(self, obs):
        """
        Rôle     : appelée une fois quand on clique Start.
        Argument : obs = mesures du robot (obs["q"] = 19 angles, obs["time"] = secondes).
        Retour   : rien. On synchronise Placo avec la VRAIE pose du robot,
                   puis on retient d'où la main part.
        """
        for i, name in enumerate(JOINTS):            # par les NOMS, pas par l'ordre de Placo
            self.model.set_joint(name, float(obs["q"][i]))
        self.model.update_kinematics()
        self.p0 = self._hand()                       # position de départ de la main
        self.t0 = obs["time"]                        # instant de départ

    def step(self, obs):
        """
        Rôle     : appelée toutes les 20 ms tant que le move tourne.
        Argument : obs = mesures du robot.
        Retour   : dictionnaire {articulation: angle en rad} pour les 3 joints du
                   bras droit. Les autres joints gardent leur dernière consigne :
                   le robot reste donc accroché comme avant.
        """
        # s va de 0 à 1 pendant duration_s, puis reste à 1
        t = obs["time"] - self.t0
        s = min(t / max(self.options["duration_s"], 0.01), 1.0)

        # La cible monte progressivement de p0 à p0 + lift_cm (z = vers le haut du tronc)
        lift = self.options["lift_cm"] / 100.0
        self.task.target_world = self.p0 + np.array([0.0, 0.0, lift * s])

        # IK : quelques petits pas vers la cible
        for _ in range(int(self.options["solver_iters"])):
            self.solver.solve(True)
            self.model.update_kinematics()

        # On lit les angles que Placo a trouvés pour le bras droit
        return {name: self.model.get_joint(name) for name in ARM_JOINTS}