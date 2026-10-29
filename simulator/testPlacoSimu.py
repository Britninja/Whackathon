import numpy as np
import placo
import microban_model
from climbing_sim import ClimbingSim, JOINTS   # JOINTS : l'ordre des 19 angles du simulateur

# Script pour comparer les angles données par la simu avec ceux donnée par placo -> ok

# ------------------------------------------------------------------
# 1. CHARGER LES DEUX "MONDES"
# ------------------------------------------------------------------
robot = placo.RobotWrapper(str(microban_model.urdf(climb=True)))   # côté Placo
sim = ClimbingSim(wall="N2", cameras=[])    # côté simulateur (cameras=[] : pas d'images, plus rapide)
sim.reset_hanging()

# Les 4 membres : nom côté simulateur -> nom de la frame côté Placo
# (la boule au bout de chaque main/pied)
LIMBS = {
    "left_hand":  "left_hand_ball",
    "right_hand": "right_hand_ball",
    "left_foot":  "left_foot_ball",
    "right_foot": "right_foot_ball",
}

# ------------------------------------------------------------------
# 2. CALCULER LA POSITION DES BOULES AVEC CHACUN
# ------------------------------------------------------------------
def placo_balls(q):
    """Positions des 4 boules selon Placo. q = 19 angles dans l'ordre JOINTS."""
    for i, name in enumerate(JOINTS):
        robot.set_joint(name, float(q[i]))   # on passe par les NOMS, pas par l'ordre de Placo
    robot.update_kinematics()                # Placo recalcule où est chaque partie
    return {limb: robot.get_T_world_frame(frame)[:3, 3].copy()
            for limb, frame in LIMBS.items()}

def sim_balls(q):
    """Positions des 4 boules selon le simulateur (cinématique directe)."""
    # On place le tronc à l'origine, sans rotation (quaternion w,x,y,z = 1,0,0,0),
    # comme Placo avec sa base à l'identité : les deux sont alors comparables.
    balls = sim.ball_positions_at(q, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    return {limb: np.array(balls[limb]) for limb in LIMBS}

# ------------------------------------------------------------------
# 3. LES POSES DE TEST
# ------------------------------------------------------------------
rng = np.random.default_rng(0)
lo, hi = sim.joint_range[:, 0], sim.joint_range[:, 1]    # limites de chaque articulation

poses = {}
poses["tout à 0"] = np.zeros(19)

q = np.zeros(19)                                          # les angles trouvés par l'IK tout à l'heure
q[JOINTS.index("right_shoulder_pitch")] = -0.379
q[JOINTS.index("right_elbow")] = 0.230
poses["bras droit levé"] = q

for k in range(3):                                        # 3 poses au hasard, dans la moitié centrale des limites
    mid, half = (lo + hi) / 2, (hi - lo) / 4
    poses[f"aléatoire {k + 1}"] = mid + rng.uniform(-1, 1, 19) * half

# ------------------------------------------------------------------
# 4. COMPARER
# ------------------------------------------------------------------
worst = 0.0
for pose_name, q in poses.items():
    p, s = placo_balls(q), sim_balls(q)
    print(f"\n--- {pose_name} ---")
    for limb in LIMBS:
        diff_mm = (p[limb] - s[limb]) * 1000              # écart Placo - simulateur, en mm
        worst = max(worst, np.linalg.norm(diff_mm))
        print(f"{limb:11s} écart (x,y,z) mm = {np.round(diff_mm, 2)}   norme = {np.linalg.norm(diff_mm):.2f} mm")

print(f"\nPLUS GRAND ÉCART : {worst:.2f} mm")
print("=> concordent" if worst < 1.0 else "=> PAS d'accord, voir les écarts ci-dessus")