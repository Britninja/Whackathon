"""MoveClimb : code de custom.py (autre équipe), repris tel quel et branché comme move (touche c)."""
import math
from pathlib import Path

import numpy as np
import placo
import microban_model
from microban_model import JOINTS


def find_urdf():
    """robot_climb.urdf à côté de ce fichier, sinon cad/robot_climb.urdf, sinon l'URDF climb de microban_model."""
    for path in (Path(__file__).with_name("robot_climb.urdf"), Path("cad/robot_climb.urdf")):
        if path.is_file():
            return str(path)
    return str(microban_model.urdf(climb=True))
LIMBS = {                           # limb name -> URDF frame that touches the wall
    "left_hand": "left_hand_ball",
    "right_hand": "right_hand_ball",
    "left_foot": "left_foot_ball",
    "right_foot": "right_foot_ball",
}

y_spacing = 0.15
z_spacing = 0.075

TOE_IN_DEG = 10.0              # how far the toes point toward the midline
TOE_IN_SIGN = -1.0             # flip sign if the feet turn the wrong way
FOOT_REST_WEIGHT = 1e-3    # keeps the planted foot's angle too (set 0.0 for the old behavior)

# Default (rest) angle per joint, in radians. The joints task softly pulls toward these.
# Any joint not listed falls back to DEFAULT_FALLBACK.
DEFAULT_FALLBACK = 0.0
DEFAULT_JOINT_ANGLES = {
    "left_hip_pitch": -2.0,
    "right_hip_pitch": -2.0,
    "left_knee": -1.0,
    "right_knee": -1.0,
    "left_shoulder_pitch": -2.0,
    "right_shoulder_pitch": -2.0,
    "left_elbow": 2.0,
    "right_elbow": 2.0,
    "left_ankle_pitch": 1.0,
    "right_ankle_pitch": 1.0,
}


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0,  1]])


class MoveClimb:
    """Climbing controller using CubicSpline3D clearance trajectories."""
    options = {}
    option_docs = {}
    variants = ("climber",)

    def _robot_to_placo_positions(self, obs):
        joint_names = self.robot.joint_names()
        for name in joint_names:
            self.robot.set_joint(name, obs["q"][JOINTS.index(name)])
        self.robot.update_kinematics()

    def _get_limb_poses(self):
        plh = self.robot.get_T_world_frame(LIMBS["left_hand"])[:3, 3].copy()
        prh = self.robot.get_T_world_frame(LIMBS["right_hand"])[:3, 3].copy()
        plf = self.robot.get_T_world_frame(LIMBS["left_foot"])[:3, 3].copy()
        prf = self.robot.get_T_world_frame(LIMBS["right_foot"])[:3, 3].copy()
        return plh, prh, plf, prf

    def _init_next_trajectory(self):
        print("------------")
        print(f"Moving limb index: {self.limb_index}")

        # Cycle through tasks: right_foot, left_foot, right_hand, left_hand
        active_task = self.tasks[self.limb_index]
        start_pos = active_task.target_world.copy()

        # Reset foot orientation weights to the rest weight by default
        self.left_foot_ori_task.configure("left_foot_ori", "soft", FOOT_REST_WEIGHT)
        self.right_foot_ori_task.configure("right_foot_ori", "soft", FOOT_REST_WEIGHT)

        # Apply orientation gain to the active foot so it stays flat
        compensation_step = [0, 0, 0]

        if active_task == self.left_foot_task:
            self.left_foot_ori_task.configure("left_foot_ori", "soft", 1.0)
            compensation_step = [0.015, 0.01, 0.025]
        elif active_task == self.right_foot_task:
            self.right_foot_ori_task.configure("right_foot_ori", "soft", 1.0)
            compensation_step = [0.015, -0.005, 0.005]
        elif active_task == self.left_hand_task:
            compensation_step = [0.04, -0.02, 0.05]
        elif active_task == self.right_hand_task:
            compensation_step = [0.12, 0.02, 0.05]

        # Build cubic trajectory for the limb that rises with clearance (pull back, lift up, place)
        waypoints = [
            [0.00, start_pos],
            [self.part_duration * 0.2, start_pos + np.array([0, 0, 0.04])],
            [self.part_duration * 0.4, start_pos + np.array([-0.03, 0, z_spacing / 2])],
            [
                self.part_duration * 0.6,
                start_pos + np.array([-0.03, 0, z_spacing * 1.6]),
            ],
            [self.part_duration * 0.8, start_pos + np.array([0, 0, z_spacing]) + compensation_step],
            [self.part_duration, start_pos + np.array([0, 0, z_spacing])],
        ]

        self.current_trajectory = placo.CubicSpline3D()
        for t, pos in waypoints:
            self.current_trajectory.add_point(t, pos, np.zeros(3))

        self.active_task = active_task

        # Advance to the next limb in the sequence
        self.limb_index = (self.limb_index + 1) % len(self.tasks)
        self.sequence_counter += 1

    def start(self, obs):
        urdf = find_urdf()
        print(f"[MoveClimb] URDF : {urdf}")
        self.robot = placo.RobotWrapper(urdf, placo.Flags.collision_as_visual)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(False)

        self._robot_to_placo_positions(obs)

        self.last_time = obs["time"]
        self.limb_index = 0
        self.movement_time = 0.0
        self.part_duration = 2  # [s] per movement segment

        self.sequence_counter = 0

        plh = np.array([0.08417194, 0.08258263, 0.16097954])
        prh = np.array([ 0.08419343, -0.08261053,  0.16094472])
        plf = np.array([ 0.09276341,  0.03737545, -0.14360264])
        prf = np.array([ 0.09277957, -0.03763053, -0.14360031])


        # Ball tasks
        self.right_foot_task = self.solver.add_position_task(LIMBS["right_foot"], prf)
        self.right_hand_task = self.solver.add_position_task(LIMBS["right_hand"], prh)
        self.left_foot_task = self.solver.add_position_task(LIMBS["left_foot"], plf)
        self.left_hand_task = self.solver.add_position_task(LIMBS["left_hand"], plh)

        self.tasks = [self.right_foot_task, self.left_foot_task, self.right_hand_task, self.left_hand_task]

        for task in self.tasks:
            task.configure("pos_task", "soft", 100.0)

        toe_in = TOE_IN_SIGN * math.radians(TOE_IN_DEG)
        # Foot orientation tasks: flat on the wall, toes angled slightly inward
        self.right_foot_ori_task = self.solver.add_orientation_task(LIMBS["right_foot"], rot_x(-toe_in))
        self.right_foot_ori_task.configure("right_foot_ori", "soft", FOOT_REST_WEIGHT)

        self.left_foot_ori_task = self.solver.add_orientation_task(LIMBS["left_foot"], rot_x(toe_in))
        self.left_foot_ori_task.configure("left_foot_ori", "soft", FOOT_REST_WEIGHT)

        # Body orientation task
        body_orientation_task = self.solver.add_orientation_task("body", np.eye(3))
        body_orientation_task.configure("body_orientation", "soft", 1e-4)

        # LIMITS & REGULARIZATION
        self.robot.set_joint_limits("left_knee", 0.01, 2.3)
        self.robot.set_joint_limits("right_knee", 0.01, 2.3)
        self.robot.set_joint_limits("left_ankle_pitch", -0.7, 0.55)
        self.robot.set_joint_limits("right_ankle_pitch", -0.7, 0.55)
        self.solver.enable_joint_limits(True)

        # Joints task: softly hold each joint at its chosen default angle
        joints = self.solver.add_joints_task()
        joints.configure("joints", "soft", 1e-5)
        joints.set_joints({
            name: DEFAULT_JOINT_ANGLES.get(name, DEFAULT_FALLBACK)
            for name in self.robot.joint_names()
        })
        self.solver.add_regularization_task(1e-3)

        # Sanity check: catch typos in DEFAULT_JOINT_ANGLES and list joints using the fallback
        unknown = set(DEFAULT_JOINT_ANGLES) - set(self.robot.joint_names())
        if unknown:
            print(f"WARNING: unknown joints in DEFAULT_JOINT_ANGLES: {unknown}")
        print("Joints without explicit default:",
              [n for n in self.robot.joint_names() if n not in DEFAULT_JOINT_ANGLES])

        # Initialize the first trajectory segment
        self._init_next_trajectory()


    def step(self, obs):
        self._robot_to_placo_positions(obs)
        now = obs["time"]
        dt = now - self.last_time
        self.last_time = now

        if dt <= 0.0:
            dt = 1e-3

        self.movement_time += dt

        # Evaluate the cubic trajectory at the current elapsed time
        if self.movement_time <= self.part_duration:
            self.active_task.target_world = self.current_trajectory.pos(self.movement_time)
        else:
            # Segment finished, start next limb's trajectory segment
            self._init_next_trajectory()
            self.movement_time = 0.0
            self.active_task.target_world = self.current_trajectory.pos(0.0)

        # Solve kinematics and integrate joint positions
        self.solver.dt = dt
        try:
            self.solver.solve(True)
        except Exception:
            self.solver.solve(False)

        # Update robot joint angles from solver output
        for name in self.robot.joint_names():
            q_val = self.robot.get_joint(name)
            v_val = self.robot.get_joint_velocity(name)
            self.robot.set_joint(name, q_val + v_val * dt)

        self.robot.update_kinematics()

        targets = {name: self.robot.get_joint(name) for name in self.robot.joint_names()}
        if self.sequence_counter >= 13*4 + 1: 
            return 
        return targets