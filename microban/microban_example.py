import time
import pinocchio as pin
import time
import placo
import numpy as np
from placo_utils.tf import tf
from placo_utils.visualization import robot_viz, robot_frame_viz, points_viz
import microban_model


mjcf_path = str(microban_model.mjcf(climb=True))
robot = placo.RobotWrapper(mjcf_path, placo.Flags.mjcf + placo.Flags.ignore_collisions)

robot.set_T_world_frame("left_foot", np.eye(4))
robot.update_kinematics()

robot.set_joint_limits("left_knee", 0.01, 2.3)
robot.set_joint_limits("right_knee", 0.01, 2.3)

T_world_leftFoot = robot.get_T_world_frame("left_foot")
T_world_rightFoot = robot.get_T_world_frame("right_foot")

solver = placo.KinematicsSolver(robot)

solver.enable_joint_limits(True)

left_foot_task = solver.add_frame_task("left_foot", T_world_leftFoot)
left_foot_task.configure("left_foot", "soft", 1e3)
right_foot_task = solver.add_frame_task("right_foot", T_world_rightFoot)
right_foot_task.configure("right_foot", "soft", 1e3)

body_task = solver.add_frame_task("body", tf.translation_matrix((0, -0.05, 0.12)))
body_task.configure("body", "soft", 1e-3, 1e-3)

right_arm_task = solver.add_position_task("right_hand_ball", np.zeros(3))
right_arm_task.configure("right_arm", "soft", 1)

R_world_rightArm = robot.get_T_world_frame("right_hand_ball")[:3, :3]
right_arm_orientation_task = solver.add_orientation_task("right_hand_ball", R_world_rightArm)
right_arm_orientation_task.configure("rightArmOrientation", "soft", 1e-3)
right_arm_orientation_task.mask.set_axises("yz")

solver.add_regularization_task(1e-6)

viz = robot_viz(robot)
t = 0
dt = 0.025

while True:
    solver.solve(True)
    #solver.dump_status()
    robot.update_kinematics()

    target_world = (0.1, -0.05 + np.cos(t*3)*0.03, 0.2 + np.sin(t*3)*0.03)
    points_viz("target", np.array([target_world]))

    right_arm_task.target_world = np.array(target_world)

    robot_frame_viz(robot, "trunk")
    for side in "left", "right":
        for limb in "foot", "hand":
            robot_frame_viz(robot, f"{side}_{limb}_ball")
            
    viz.display(robot.state.q)
    time.sleep(dt)
    t += dt
