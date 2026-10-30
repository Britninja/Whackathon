import numpy as np
import placo
import microban_model
from microban_model import JOINTS

# Ce script sert à bouger la boule du bras droit du robot vers le haut, avec des parametres ajustables
# Pour cela, il utilise un solver placo qui update au fur et a mesure la position cible de cette boule vers la position demandé

class BasePlacoMove:
    options = {"dx_cm": 0.0, "dy_cm": 0.0, "dz_cm": 3.0, "duration_s": 2.0, "solver_iters": 5}    
    
    # Ces attributs doivent être définis par les classes enfants
    FRAME = None 
    MEMBER_JOINTS = None

    def __init__(self):
        """
        Rôle    : prépare Placo UNE SEULE FOIS, au démarrage du programme.
                    (Charger le modèle est lent : on ne le fait jamais dans step.)
        Attention : on appelle nos objets self.model / self.solver, jamais
                    self.robot (ce nom est réservé par le programme de microban).
        """
        self.model = placo.RobotWrapper(str(microban_model.urdf(climb=True)))
        self.solver = placo.KinematicsSolver(self.model)

        self.solver.mask_fbase(True)  # Tronc fixe
        
        self.model.update_kinematics()

    def _current_pos(self):
        """
        Rôle    : position actuelle de la main dans le modèle Placo.
        Retour  : tableau numpy (x, y, z) en mètres, repère du tronc.
        """
        return self.model.get_T_world_frame(self.FRAME)[:3, 3].copy()

    def start(self, obs):
        """
        Rôle     : appelée une fois quand on clique Start.
        Argument : obs = mesures du robot (obs["q"] = 19 angles, obs["time"] = secondes).
        Retour   : rien. On synchronise Placo avec la VRAIE pose du robot,
                puis on retient d'où la main part.
        """
        for i, name in enumerate(JOINTS):
            self.model.set_joint(name, float(obs["q"][i]))
        self.model.update_kinematics()
        self.p0 = self._current_pos()
        self.t0 = obs["time"]

    def step(self, obs):
        """
        Rôle     : appelée toutes les 20 ms tant que le move tourne.
        Argument : obs = mesures du robot.
        Retour   : dictionnaire {articulation: angle en rad} pour les 3 joints du
                    bras droit. Les autres joints gardent leur dernière consigne :
                    le robot reste donc accroché comme avant.
        """
        t = obs["time"] - self.t0
        s = min(t / max(self.options["duration_s"], 0.01), 1.0)

        # Conversion des centimètres en mètres pour chaque axe
        dx = self.options["dx_cm"] / 100.0
        dy = self.options["dy_cm"] / 100.0
        dz = self.options["dz_cm"] / 100.0

        # Application du déplacement sur X, Y et Z de manière lissée par s
        self.task.target_world = self.p0 + np.array([dx * s, dy * s, dz * s])

        for _ in range(int(self.options["solver_iters"])):
            self.solver.solve(True)
            self.model.update_kinematics()

        return {name: self.model.get_joint(name) for name in self.MEMBER_JOINTS}


# --- Déclinaisons simples pour chaque membre ---

class MoveRightArm(BasePlacoMove):
    """Fait bouger la main droite en X, Y, Z."""
    variants = ("climber",)
    FRAME = "right_hand_ball"
    MEMBER_JOINTS = (
        "right_shoulder_pitch", 
        "right_shoulder_roll", 
        "right_elbow"
    )
    option_docs = {
        "dx_cm": "déplacement X (cm)", 
        "dy_cm": "déplacement Y (cm)", 
        "dz_cm": "déplacement Z (cm)", 
        "duration_s": "durée (s)", 
        "solver_iters": "itérations"
    }

class MoveLeftArm(BasePlacoMove):
    """Fait bouger la main gauche en X, Y, Z."""
    variants = ("climber",)
    FRAME = "left_hand_ball"
    MEMBER_JOINTS = (
        "left_shoulder_pitch", 
        "left_shoulder_roll", 
        "left_elbow"
    )
    option_docs = {
        "dx_cm": "déplacement X (cm)", 
        "dy_cm": "déplacement Y (cm)", 
        "dz_cm": "déplacement Z (cm)", 
        "duration_s": "durée (s)", 
        "solver_iters": "itérations"
    }

class MoveRightLeg(BasePlacoMove):
    """Fait bouger le pied droit en X, Y, Z."""
    variants = ("climber",)
    FRAME = "right_foot_ball"
    MEMBER_JOINTS = (
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle_pitch",
        "right_ankle_roll"
    )
    option_docs = {
        "dx_cm": "déplacement X (cm)", 
        "dy_cm": "déplacement Y (cm)", 
        "dz_cm": "déplacement Z (cm)", 
        "duration_s": "durée (s)", 
        "solver_iters": "itérations"
    }

class MoveLeftLeg(BasePlacoMove):
    """Fait bouger le pied gauche en X, Y, Z."""
    variants = ("climber",)
    FRAME = "left_foot_ball"
    MEMBER_JOINTS = (
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll"
    )
    option_docs = {
        "dx_cm": "déplacement X (cm)", 
        "dy_cm": "déplacement Y (cm)", 
        "dz_cm": "déplacement Z (cm)", 
        "duration_s": "durée (s)", 
        "solver_iters": "itérations"
    }