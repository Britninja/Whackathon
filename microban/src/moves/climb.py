import json
from pathlib import Path
import time

import numpy as np
import placo
import microban_model
from placo_utils.visualization import robot_viz, points_viz
from simulator.climbing_sim import JOINTS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAP_PATH = PROJECT_ROOT / "simulator/climbing_sim/walls" / "N1.json"

with open(MAP_PATH, "r") as f:
    _HOLDS_MAP = json.load(f)

# TODO: Are these world or robot positions
_START_HOLDS = {
    "right_foot": _HOLDS_MAP["holds"][0],
    "left_foot": _HOLDS_MAP["holds"][1],
    "right_hand": _HOLDS_MAP["holds"][2],
    "left_hand": _HOLDS_MAP["holds"][3],
}

MJCF_PATH = str(microban_model.mjcf(climb=True))


class ClimbMove:
    start_pose = "current"
    options = {"y_spacing": 0.15, "z_spacing": 0.06}
    option_docs = {
        "y_spacing": "Distance left foot (to right foot)",
        "z_spacing": "hopefully hold_distance (rows)",
    }
    variants = ("climber",)

    def __init__(self) -> None:
        # "Holds" position (just for visualization)
        self._vis_holds = [
            [0, y, z]
            for y in [0, self.options["y_spacing"]]
            for z in np.arange(0, 1, self.options["z_spacing"])
        ]

        self._robot = placo.RobotWrapper(
            MJCF_PATH, placo.Flags.mjcf + placo.Flags.ignore_collisions
        )
        self._solver = placo.KinematicsSolver(self._robot)

        # Set left foot as world origin
        self._T_left_foot = np.eye(4)
        self._robot.set_T_world_frame("left_foot", self._T_left_foot)
        self._robot.update_kinematics()

        # Set center of mass to robot center of mass (in world coordinates)
        # self._com_initial = self._robot.com_world()
        T_right_foot = self._robot.get_T_world_frame("right_foot").copy()
        T_right_foot[2, 3] = 0.0

        # Limit knee angles
        self._robot.set_joint_limits("left_knee", 0.01, 2.3)
        self._robot.set_joint_limits("right_knee", 0.01, 2.3)

        # Ball tasks
        right_foot_task = self._solver.add_position_task(
            "right_foot_ball", np.array([0, 0, 0])
        )
        left_foot_task = self._solver.add_position_task(
            "left_foot_ball", np.array([0, self.options["y_spacing"], 0])
        )
        right_hand_task = self._solver.add_position_task(
            "right_hand_ball", np.array([0, 0, self.options["z_spacing"] * 5])
        )
        left_hand_task = self._solver.add_position_task(
            "left_hand_ball",
            np.array([0, self.options["y_spacing"], self.options["z_spacing"] * 5]),
        )

        self._tasks = [right_foot_task, left_foot_task, right_hand_task, left_hand_task]

        # Keep trunk parallel to the world
        body_orientation_task = self._solver.add_orientation_task("body", np.eye(3))
        body_orientation_task.configure("body_orientation", "soft", 1e-3)

        # Joints -> keep them close to their zero position
        joints_task = self._solver.add_joints_task()
        joints_task.configure("joints", "soft", 1e-6)

        joints_task.set_joints({joint: 0 for joint in JOINTS})

        # TODO: is this for not moving the body much in general?
        self._solver.add_regularization_task(1e-3)
        self._solver.enable_joint_limits(True)

        # TODO: is this for calculating the neutral position ???
        for _ in range(50):
            self._solver.solve(True)
            self._robot.update_kinematics()

        self._start_target_angles = [
            self._robot.get_joint(name) for name in JOINTS
        ]
        self._start_angles = self._start_target_angles.copy()
        self._task_rotation_index = 0

    def start(self, obs) -> None:
        self.t0 = obs["time"]
        self.stop_t0 = None
        for name, angle in zip(JOINTS, self._start_angles):
            self._robot.set_joint(name, angle)
        self._robot.set_T_world_frame("left_foot", self._T_left_foot)
        self._robot.update_kinematics()

    def calc_4_waypoints(self, task, part_duration):
        return [
            [0, task.target_world],
            [part_duration * 0.25, task.target_world + np.array([-0.05, 0, 0])],
            [
                part_duration * 0.5,
                task.target_world + np.array([-0.05, 0, self.options["z_spacing"]]),
            ],
            [
                part_duration * 0.75,
                task.target_world
                + np.array([-0.025, 0, self.options["z_spacing"] + 0.02]),
            ],
            [
                part_duration,
                task.target_world + np.array([0, 0, self.options["z_spacing"]]),
            ],
        ]  # [time, (x,y,z)]

    def step(self, obs):
        viz = robot_viz(self._robot)
        dt = 0.02  # [s]
        part_duration = 1.5  # [s]
        # TODO: maybe axis task: try to avoid collision of ball with the wall of the hold
        # TODO: parameterize, so that if a hold moves 1cm => make it so that you can reach anywhere the arms can reach => inverse kinematics
        # Problem is a big space & there is only one solution
        # <=> inverse kinematics = underdefined: there are many solutions, you don't know which one is the best
        # kinematics = only geometry, not forces/torques
        # => we migt plan a good ttrajectory, but the moteurs might not be strong enough
        # Building a cubic trajectory for the limb that rise with clearance
        task = self._tasks[self._task_rotation_index]
        waypoints = [
            [0, task.target_world],
            [part_duration * 0.25, task.target_world + np.array([-0.05, 0, 0])],
            [
                part_duration * 0.5,
                task.target_world + np.array([-0.05, 0, self.options["z_spacing"]]),
            ],
            [
                part_duration * 0.75,
                task.target_world
                + np.array([-0.025, 0, self.options["z_spacing"] + 0.02]),
            ],
            [
                part_duration,
                task.target_world + np.array([0, 0, self.options["z_spacing"]]),
            ],
        ]  # [time, (x,y,z)]

        trajectory = placo.CubicSpline3D()
        velocity_3d = np.zeros(3)
        for move_duration_t, pos_3d in waypoints:
            trajectory.add_point(move_duration_t, pos_3d, velocity_3d)

        # move_duration_t = 0
        part_t0 = time.time()
        while time.time() - part_t0 < part_duration:
            task.target_world = trajectory.pos(time.time() - part_t0)

            self._solver.solve(True)
            self._robot.update_kinematics()

            points_viz("holds", self._vis_holds)

            viz.display(self._robot.state.q)
            time.sleep(dt)
            # move_duration_t += dt
        self._task_rotation_index = (self._task_rotation_index + 1) % len(self._tasks)

        return {name: self._robot.get_joint(name) for name in JOINTS}
