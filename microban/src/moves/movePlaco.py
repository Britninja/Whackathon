import numpy as np
import placo
import microban_model
from microban_model import JOINTS

FRAMES = ("right_foot_ball", "left_foot_ball", "right_hand_ball", "left_hand_ball")

class BasePlacoMove:
    options = {"dx_cm": 0.0, "dy_cm": 0.0, "dz_cm": 3.0, "duration_s": 2.0, "solver_iters": 5}
    option_docs = {
        "dx_cm": "déplacement X (cm)",
        "dy_cm": "déplacement Y (cm)",
        "dz_cm": "déplacement Z (cm)",
        "duration_s": "durée (s)",
        "solver_iters": "itérations",
    }
    variants = ("climber",)

    FRAME = None  # défini par les classes enfants : le membre qui bouge

    # Modèle et solveur PARTAGÉS par tous les mouvements : le tronc reste en mémoire
    model = None
    solver = None
    tasks = None

    def __init__(self):
        """Prépare Placo une seule fois pour tous les mouvements."""
        cls = BasePlacoMove # Partage des mouvements entre les 4 membres
        if cls.model is None:
            cls.model = placo.RobotWrapper(str(microban_model.urdf(climb=True)))
            cls.solver = placo.KinematicsSolver(cls.model)  # tronc LIBRE (pas de mask_fbase)

            cls.model.set_joint_limits("left_knee", 0.01, 2.3)
            cls.model.set_joint_limits("right_knee", 0.01, 2.3)

            # Une tâche de position par membre
            cls.tasks = {f: cls.solver.add_position_task(f, np.zeros(3)) for f in FRAMES}

            body_orientation_task = cls.solver.add_orientation_task("body", np.eye(3))
            body_orientation_task.configure("body_orientation", "soft", 1e-3)

            joints = cls.solver.add_joints_task()
            joints.configure("joints", "soft", 1e-6)
            joints.set_joints({j: 0 for j in cls.model.joint_names()})

            cls.solver.add_regularization_task(1e-3)
            cls.solver.enable_joint_limits(True)
            cls.model.update_kinematics()

    def _current_pos(self):
        return self.model.get_T_world_frame(self.FRAME)[:3, 3].copy()

    def start(self, obs):
        # Synchronise les articulations (le tronc reste celui du modèle partagé)
        for i, name in enumerate(JOINTS):
            self.model.set_joint(name, float(obs["q"][i]))
        self.model.update_kinematics()

        # Chaque membre vise sa position actuelle : il reste où il est
        for f in FRAMES:
            self.tasks[f].target_world = self.model.get_T_world_frame(f)[:3, 3].copy()

        self.p0 = self._current_pos()
        self.t0 = obs["time"]

    # Constantes du mouvement de base
    LIFT = 0.015   # levée initiale ET dépassement (même valeur)
    BACK = 0.04    # recul en X

    def _path(self, s):
        """Décalage (m) par rapport à p0, s va de 0 à 1."""
        o = self.options
        dx, dy, dz = o["dx_cm"] / 100.0, o["dy_cm"] / 100.0, o["dz_cm"] / 100.0
        L, B = self.LIFT, self.BACK

        # Les 6 points du chemin, parcourus à vitesse régulière
        pts = np.array([
            [0.0,    0.0, 0.0],
            [0.0,    0.0, L],         # se lève
            [dx - B, dy,  L],         # recule
            [dx - B, dy,  dz + L],    # monte un peu plus haut que le point
            [dx,     dy,  dz + L],    # avance jusqu'au x
            [dx,     dy,  dz],        # redescend sur le z
        ])

        spline = placo.CubicSpline3D()
        for k, p in enumerate(pts):
            spline.add_point(k / (len(pts) - 1), np.array(p), np.zeros(3))  # vitesse nulle à chaque point

        return spline.pos(s)
    
    def step(self, obs):
        s = min((obs["time"] - self.t0) / max(self.options["duration_s"], 0.01), 1.0)
        self.tasks[self.FRAME].target_world = self.p0 + self._path(s)

        for _ in range(int(self.options["solver_iters"])):
            self.solver.solve(True)
            self.model.update_kinematics()

        # On renvoie tous les joints : le tronc monte, les jambes se tendent
        return {name: self.model.get_joint(name) for name in JOINTS}

# --- Déclinaisons simples pour chaque membre ---

class MoveRightArm(BasePlacoMove):
    """Fait bouger la main droite en X, Y, Z."""
    FRAME = "right_hand_ball"

class MoveLeftArm(BasePlacoMove):
    """Fait bouger la main gauche en X, Y, Z."""
    FRAME = "left_hand_ball"

class MoveRightLeg(BasePlacoMove):
    """Fait bouger le pied droit en X, Y, Z."""
    FRAME = "right_foot_ball"
    
class MoveLeftLeg(BasePlacoMove):
    """Fait bouger le pied gauche en X, Y, Z."""
    FRAME = "left_foot_ball"
