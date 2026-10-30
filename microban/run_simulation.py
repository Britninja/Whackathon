import time
import numpy as np
import placo
import microban_model
from microban_model import JOINTS
from placo_utils.visualization import robot_viz, points_viz

from src.moves.movePlaco import (
    MoveRightArm, MoveLeftArm, MoveRightLeg, MoveLeftLeg, BasePlacoMove,
)

y_spacing = 0.15
z_spacing = 0.06
holds = [[0, y, z] for y in [0, y_spacing] for z in np.arange(0, 1, z_spacing)]
dt = 0.02


def build_start_pose():
    """Pose d'escalade initiale : les 4 membres sur les prises, tronc libre, pieds parallèles."""
    robot = placo.RobotWrapper(
        str(microban_model.mjcf(climb=True)),
        placo.Flags.mjcf + placo.Flags.ignore_collisions,
    )
    robot.update_kinematics()
    R_rf = robot.get_T_world_frame("right_foot_ball")[:3, :3].copy()
    R_lf = robot.get_T_world_frame("left_foot_ball")[:3, :3].copy()

    solver = placo.KinematicsSolver(robot)
    robot.set_joint_limits("left_knee", 0.01, 2.3)
    robot.set_joint_limits("right_knee", 0.01, 2.3)

    solver.add_position_task("right_foot_ball", np.array([0, 0, 0]))
    solver.add_position_task("left_foot_ball", np.array([0, y_spacing, 0]))
    solver.add_position_task("right_hand_ball", np.array([0, 0, 5 * z_spacing]))
    solver.add_position_task("left_hand_ball", np.array([0, y_spacing, 5 * z_spacing]))

    for frame, R in (("right_foot_ball", R_rf), ("left_foot_ball", R_lf)):
        t = solver.add_orientation_task(frame, R)
        t.configure(f"{frame}_orientation", "soft", 1.0)

    body = solver.add_orientation_task("body", np.eye(3))
    body.configure("body_orientation", "soft", 1e-3)
    j = solver.add_joints_task()
    j.configure("joints", "soft", 1e-6)
    j.set_joints({n: 0 for n in robot.joint_names()})
    solver.add_regularization_task(1e-3)
    solver.enable_joint_limits(True)

    for _ in range(100):
        solver.solve(True)
        robot.update_kinematics()
    return robot


def limb_cycle(MoveClass, rise_cm=6.0, T=2.0):
    """Un pas complet : lever, reculer, monter, avancer, redescendre."""
    return [
        (MoveClass, {"dx_cm": 0.0, "dy_cm": 0.0, "dz_cm": rise_cm, "duration_s": T}),
    ]


def main():
    sim_robot = build_start_pose()
    viz = robot_viz(sim_robot)

    # Les Move* partagent un seul modèle Placo, créé dans __init__ (lent : une seule fois)
    moves = {cls: cls() for cls in (MoveRightLeg, MoveLeftLeg, MoveRightArm, MoveLeftArm)}

    # Une seule fois : le modèle partagé prend le tronc de la pose de départ
    BasePlacoMove.model.set_T_world_fbase(sim_robot.get_T_world_fbase())

    # État des 19 articulations issu de la pose d'escalade
    q_current = np.array([sim_robot.get_joint(n) for n in JOINTS], dtype=float)

    sequence = (
        limb_cycle(MoveRightLeg)
        + limb_cycle(MoveLeftLeg)
        + limb_cycle(MoveRightArm)
        + limb_cycle(MoveLeftArm)
    )

    sim_time = 0.0
    print("Lancement de la simulation d'escalade...")

    points_viz("holds", holds)

    while True:
        for MoveClass, options in sequence:
            move = moves[MoveClass]
            move.options.update(options)

            obs = {"time": sim_time, "q": q_current.copy()}
            move.start(obs)

            n_steps = int(round(options["duration_s"] / dt)) + 1

            for _ in range(n_steps):
                t_loop = time.time()                                  # <-- AJOUT

                obs["time"] = sim_time
                target_joints = move.step(obs)

                for name, angle in target_joints.items():
                    sim_robot.set_joint(name, angle)
                    q_current[JOINTS.index(name)] = angle

                sim_robot.set_T_world_fbase(BasePlacoMove.model.get_T_world_fbase())
                sim_robot.update_kinematics()

                viz.display(sim_robot.state.q)                        # points_viz supprimé d'ici

                time.sleep(max(0.0, dt - (time.time() - t_loop)))     # <-- REMPLACE time.sleep(dt)
                sim_time += dt


if __name__ == "__main__":
    main()