import math
import numpy as np
import placo
import microban_model
from microban_model import JOINTS

FRAMES = ("right_foot_ball", "left_foot_ball", "right_hand_ball", "left_hand_ball")
FEET = ("right_foot_ball", "left_foot_ball")

# RÉGLAGES

# Trajectoire d'un membre (m) : se lève de LIFT, recule de BACK, monte jusqu'à OVER au-dessus
# de la prise, avance en appuyant de PRESS vers le mur, puis se pose sur la prise.
LIFT = 0.04
BACK = 0.04
OVER = {"hand": 0.05, "foot": 0.03}    # hauteur de passage au-dessus de la prise
PRESS = {"hand": 0.05, "foot": 0.03}  # appui vers le mur à l'approche (viser un peu dans le
                                       # mur crée une force qui accroche la boule à la prise)

# Posture de repos = pose de départ + ces préférences (rad, mêmes valeurs à gauche et à droite)
REST_BIAS = {
    "hip_pitch": -0.4,    # fléchir les hanches
    "elbow": 0.8,         # plier les coudes
    "ankle_pitch": 0.5,   # monter la pointe du pied
}

# Pieds à plat sur le mur, presque parallèles
TOE_IN_DEG = 5.0          # pointes vers l'intérieur (0 = parallèles, signe opposé si ça tourne à l'envers)

def rot_z(angle):
    """Rotation autour de la verticale."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

class BasePlacoMove:
    options = {"dx_cm": 0.0, "dy_cm": 0.0, "dz_cm": 7.5, "duration_s": 3.0}
    option_docs = {
        "dx_cm": "déplacement X (cm)",
        "dy_cm": "déplacement Y (cm)",
        "dz_cm": "déplacement Z (cm)",
        "duration_s": "durée (s)",
    }
    variants = ("climber",)

    FRAME = None  # défini par les classes enfants : le membre qui bouge

    # Modèle et solveur PARTAGÉS par tous les mouvements : le tronc reste en mémoire
    model = None
    solver = None
    tasks = None
    foot_ori = None
    joints_task = None
    initialized = False  # la pose de départ a-t-elle déjà été lue ?

    def __init__(self):
        """Prépare Placo une seule fois pour tous les mouvements."""
        self.done = False
        cls = BasePlacoMove # Partage des mouvements entre les 4 membres
        if cls.model is None:
            cls.model = placo.RobotWrapper(str(microban_model.urdf(climb=True)))
            cls.solver = placo.KinematicsSolver(cls.model)  # tronc LIBRE (pas de mask_fbase)

            # Limites : genoux, et chevilles (le pied ne se casse pas)
            cls.model.set_joint_limits("left_knee", 0.01, 2.3)
            cls.model.set_joint_limits("right_knee", 0.01, 2.3)
            cls.model.set_joint_limits("left_ankle_pitch", -0.7, 0.55)
            cls.model.set_joint_limits("right_ankle_pitch", -0.7, 0.55)

            # Tâches de position : chaque boule vise sa prise (poids fort, c'est la priorité)
            cls.tasks = {f: cls.solver.add_position_task(f, np.zeros(3)) for f in FRAMES}
            for f, task in cls.tasks.items():
                task.configure(f"{f}_pos", "soft", 100.0)

            # Tâches d'orientation des pieds : à plat, fortes pour le pied qui bouge
            toe = math.radians(TOE_IN_DEG)
            cls.foot_ori = {
                "right_foot_ball": cls.solver.add_orientation_task("right_foot_ball", rot_z(toe)),
                "left_foot_ball": cls.solver.add_orientation_task("left_foot_ball", rot_z(-toe)),
            }
            for f, task in cls.foot_ori.items():
                task.configure(f"{f}_ori", "soft", 1e-3)

            # Tâche d'orientation du tronc : reste à peu près droit
            body_orientation_task = cls.solver.add_orientation_task("body", np.eye(3))
            body_orientation_task.configure("body_orientation", "soft", 1e-3)

            # Tâche articulaire : tire doucement vers la posture de repos (fixée au 1er start)
            cls.joints_task = cls.solver.add_joints_task()
            cls.joints_task.configure("joints", "soft", 1e-5)
            cls.joints_task.set_joints({j: 0.0 for j in cls.model.joint_names()})

            # Régularisation : évite les mouvements brusques
            cls.solver.add_regularization_task(1e-3)
            cls.solver.enable_joint_limits(True)
            cls.model.update_kinematics()

    def _sync(self, obs):
        """Recopie les angles mesurés dans le modèle Placo (le tronc reste celui du modèle)."""
        for i, name in enumerate(JOINTS):
            self.model.set_joint(name, float(obs["q"][i]))
        self.model.update_kinematics()

    def start(self, obs):
        self._sync(obs)

        # Au premier mouvement, la pose de départ sert de référence :
        # cibles des boules et posture de repos (pose de départ + REST_BIAS)
        cls = BasePlacoMove
        if not cls.initialized:
            for f in FRAMES:
                self.tasks[f].target_world = self.model.get_T_world_frame(f)[:3, 3].copy()
            rest = {}
            for j in self.model.joint_names():
                bias = REST_BIAS.get(j.replace("left_", "").replace("right_", ""), 0.0)
                rest[j] = self.model.get_joint(j) + bias
            self.joints_task.set_joints(rest)
            cls.initialized = True

        # Le pied qui bouge reste à plat, l'autre seulement un peu
        for f, task in self.foot_ori.items():
            task.configure(f"{f}_ori", "soft", 1.0 if f == self.FRAME else 1e-3)

        # Le membre part de sa dernière cible (pas de dérive d'un mouvement à l'autre)
        self.p0 = self.tasks[self.FRAME].target_world.copy()
        self.t0 = obs["time"]
        self.last_time = obs["time"]
        self.done = False

        # Membre fixe de référence pour le bilan : une main pour un pied, un pied pour une main
        self.ref = "right_hand_ball" if self.FRAME in FEET else "right_foot_ball"
        self.reported = False

    def _path(self, s):
        """Décalage (m) par rapport à p0, s va de 0 à 1."""
        o = self.options
        dx, dy, dz = o["dx_cm"] / 100.0, o["dy_cm"] / 100.0, o["dz_cm"] / 100.0
        limb = "foot" if self.FRAME in FEET else "hand"
        H, P = OVER[limb], PRESS[limb]

        # Les 6 points du chemin, parcourus à vitesse régulière
        pts = np.array([
            [0.0,         0.0, 0.0],
            [0.0,         0.0, LIFT],     # se lève
            [dx - BACK,   dy,  LIFT],     # recule
            [dx - BACK,   dy,  dz + H],   # monte au-dessus de la prise
            [dx + P,      dy,  dz + H],   # avance en appuyant vers le mur
            [dx,          dy,  dz],       # se pose sur la prise
        ])

        spline = placo.CubicSpline3D()
        for k, p in enumerate(pts):
            spline.add_point(k / (len(pts) - 1), np.array(p), np.zeros(3))  # vitesse nulle à chaque point

        return spline.pos(s)

    def step(self, obs):
        s = min((obs["time"] - self.t0) / max(self.options["duration_s"], 0.01), 1.0)
        self.tasks[self.FRAME].target_world = self.p0 + self._path(s)
        self.done = s >= 1.0

        # On repart des angles mesurés à chaque pas, puis un seul pas de solveur :
        # si un membre s'affaisse, l'erreur est vue au pas suivant et corrigée
        self._sync(obs)

        # Bilan en fin de mouvement (le modèle contient les angles mesurés à ce moment-là)
        if self.done and not self.reported:
            self._report()
            self.reported = True

        dt = obs["time"] - self.last_time
        self.last_time = obs["time"]
        if dt <= 0.0:
            dt = 1e-3
        self.solver.dt = dt
        try:
            self.solver.solve(True)
        except Exception:
            self.solver.solve(False)
        for name in self.model.joint_names():
            self.model.set_joint(name, self.model.get_joint(name) + self.model.get_joint_velocity(name) * dt)
        self.model.update_kinematics()

        # On renvoie tous les joints : le tronc monte, les jambes se tendent
        return {name: self.model.get_joint(name) for name in JOINTS}

    def _report(self):
        """Écart (cm) entre la vraie boule et sa prise, mesuré par rapport au membre de référence
        (ne dépend pas de la position du tronc)."""
        T = self.model.get_T_world_frame
        measured = T(self.FRAME)[:3, 3] - T(self.ref)[:3, 3]
        wanted = self.tasks[self.FRAME].target_world - self.tasks[self.ref].target_world
        err = 100 * (measured - wanted)
        print(f"[bilan] {self.FRAME:16s} écart (cm) : mur {err[0]:+5.1f}  côté {err[1]:+5.1f}  haut {err[2]:+5.1f}")

# --- Déclinaisons simples pour chaque membre ---

class MoveRightArm(BasePlacoMove):
    """Fait bouger la main droite en X, Y, Z."""
    variants = ("climber",)
    FRAME = "right_hand_ball"

class MoveLeftArm(BasePlacoMove):
    """Fait bouger la main gauche en X, Y, Z."""
    variants = ("climber",)
    FRAME = "left_hand_ball"

class MoveRightLeg(BasePlacoMove):
    """Fait bouger le pied droit en X, Y, Z."""
    variants = ("climber",)
    FRAME = "right_foot_ball"

class MoveLeftLeg(BasePlacoMove):
    """Fait bouger le pied gauche en X, Y, Z."""
    variants = ("climber",)
    FRAME = "left_foot_ball"