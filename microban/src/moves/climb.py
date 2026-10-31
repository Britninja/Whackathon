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


hole_dist = 0.0014  # m
ball_rad = 0.00125  # m
ball_center_wall_dist = 0.0026  # m
wall_start_leg_dist = 5 * hole_dist  # m
wall_start_arm_dist = 11 * hole_dist  # m
wall_start_arm_leg_dist = 3 * hole_dist  # m (left-left or right-right)
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

        # left_foot_start coordinates
        # x = towards the wall?, y left right, z hight
        self.left_foot_start = [0.0] * 3
        self.right_foot_start = [0.0] * 3
        self.right_hand_start = [0.0] * 3
        self.left_hand_start = [0.0] * 3
     
        self.left_foot_start[0] = -ball_center_wall_dist
        self.left_foot_start[1] = _START_HOLDS["left_foot"]["col"] * hole_dist
        self.left_foot_start[2] = _START_HOLDS["left_foot"]["row"] * hole_dist
     
        self.right_foot_start[0] = -ball_center_wall_dist
        self.right_foot_start[1] = _START_HOLDS["right_foot"]["col"] * hole_dist
        self.right_foot_start[2] = _START_HOLDS["right_foot"]["row"] * hole_dist
     
        self.right_hand_start[0] = -ball_center_wall_dist
        self.right_hand_start[1] = _START_HOLDS["right_hand"]["col"] * hole_dist
        self.right_hand_start[2] = _START_HOLDS["right_hand"]["row"] * hole_dist
     
        self.left_hand_start[0] = -ball_center_wall_dist
        self.left_hand_start[1] = _START_HOLDS["left_hand"]["col"] * hole_dist
        self.left_hand_start[2] = _START_HOLDS["left_hand"]["row"] * hole_dist

        print()

        # Set left foot as world origin
        # self._T_left_foot = np.eye(4)
        # self._robot.set_T_world_frame("left_foot", self._T_left_foot)
        # self._robot.update_kinematics()

        # Set center of mass to robot center of mass (in world coordinates)
        # self._com_initial = self._robot.com_world()

        # T_right_foot = self._robot.get_T_world_frame("right_foot").copy()
        # T_right_foot[2, 3] = 0.0

        # Limit knee angles
        self._robot.set_joint_limits("left_knee", 0.01, 2.3)
        self._robot.set_joint_limits("right_knee", 0.01, 2.3)

        # Ball tasks
        right_foot_task = self._solver.add_position_task(
            "right_foot_ball", np.array(self.right_foot_start)
        )
        left_foot_task = self._solver.add_position_task(
            "left_foot_ball", np.array(self.left_foot_start)
        )
        right_hand_task = self._solver.add_position_task(
            "right_hand_ball", np.array(self.right_hand_start)
        )
        left_hand_task = self._solver.add_position_task(
            "left_hand_ball", np.array(self.left_hand_start)
        )
        self._ball_joints = [
            "right_foot_ball",
            "left_foot_ball",
            "right_hand_ball",
            "left_hand_ball",
        ]
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
        # for _ in range(50):
        #     self._solver.solve(True)
        #     self._robot.update_kinematics()

        # self._start_target_angles = [self._robot.get_joint(name) for name in JOINTS]
        # self._start_angles = self._start_target_angles.copy()
        # self._task_rotation_index = 0

    def start(self, obs) -> None:
        # self.t0 = obs["time"]
        # self.time_elapsed_since_last_step = 0
        # self.stop_t0 = None
        # for name, angle in zip(JOINTS, self._start_angles):
        #     self._robot.set_joint(name, angle)
        # self._robot.set_T_world_frame("left_foot", self._T_left_foot)
        # self._robot.update_kinematics()
        self._did_move = False
        print("start climb")

    def calc_target_update_waypoints(self, task, part_duration):
        prev_target = task.target_world  # Start trajectory at the old target
        # xd = -0.05  # through wall
        # yd = 0  # horizontally, left right
        # zd = 0.02  # height

        # define a wall point => transform the
        # x = rouge dans le mur; y gauche vert, bleue z,  (frame viz with identity (origin))

        x = [
            0, # 0
            0, # 1
            -2 * ball_center_wall_dist, # 2
            -2 * ball_center_wall_dist, # 3
            0, # 4
            0, # 5
            # 6
        ]  # x: into wall (- is away from wall)
        y = [0, 0, 0, 0, 0, 0]  # y: gauche
        z = [0, 3 * hole_dist, 3 * hole_dist, 6 * hole_dist, 6 * hole_dist, 5 * hole_dist]  # height
        n_steps = len(x)
        steps = [None] * n_steps
        for i in range(n_steps):
            steps[i] = [
                (i / (n_steps - 1)) * part_duration,
                prev_target + [x[i], y[i], z[i]],
            ]
        return steps
        # return [
        #     [0 * part_duration, prev_target],
        #     [0.25 * part_duration, prev_target + [xs[0], ys[0], zs[0]]],
        #     [0.5 * part_duration, prev_target + [0, ys[1], zs[1]]],
        #     [0.75 * part_duration, prev_target + [0, ys[2], zs[2]]],
        #     [1 * part_duration, prev_target + [0, ys[3], zs[3]]],
        # ]

        # return [
        #     [0, prev_target],
        #     [part_duration * 0.2, prev_target + np.array([xd, 0, zd])],
        #     [
        #         part_duration * 0.4,
        #         prev_target + np.array([xd, 0, zd]),
        #     ],
        #     [
        #         part_duration * 0.6,
        #         prev_target + np.array([xd, 0, self.options["z_spacing"] + zd]),
        #     ],
        #     [
        #         part_duration * 0.8,
        #         prev_target + np.array([0, 0, self.options["z_spacing"] + zd]),
        #     ],            [
        #         part_duration,
        #         prev_target + np.array([0, 0, self.options["z_spacing"]]),
        #     ],
        # ]  # [time, (x,y,z)]

    def step(self, obs):
        return self._move_task(obs, self._tasks[2])


    def _move_task(self, obs, task):
        if not self._did_move: 
            viz = robot_viz(self._robot)
            dt = 0.02  # [s]
            part_duration = 1.5  # [s]
            waypoints = self.calc_target_update_waypoints(task, part_duration)
            trajectory = placo.CubicSpline3D()
            for t, pos in waypoints:
                trajectory.add_point(t, pos, np.zeros(3))

            t = 0
            t0 = time.time()
            while time.time() - t0 < part_duration:
                task.target_world = trajectory.pos(time.time() - t0)

                self._solver.solve(True)
                self._robot.update_kinematics()

                points_viz("holds", self._vis_holds)

                viz.display(self._robot.state.q)
                # time.sleep(dt)
                # t += dt
            self._did_move = True
            return {name: self._robot.get_joint(name) for name in JOINTS}
        return {}

    def _cycle_moves(self, obs):
        for task in self._tasks:
            self._move_task(obs, task)
        return {name: self._robot.get_joint(name) for name in JOINTS}


    def _climb(self, obs):
        viz = robot_viz(self._robot)
        part_duration = 6  # [s]
        print(f"Start climb: {time.time()}")

        for i, task in enumerate(self._tasks):
            ball = self._ball_joints[i]

            print(f"task: {ball} at {time.time()}")

            waypoints = self.calc_target_update_waypoints(task, part_duration)

            trajectory = placo.CubicSpline3D()
            velocity_3d = np.zeros(3)
            for move_time_t, pos_3d in waypoints:
                trajectory.add_point(move_time_t, pos_3d, velocity_3d)

            t0 = time.time()
            while time.time() - t0 < part_duration:
                print("update target")
                task.target_world = trajectory.pos(time.time() - t0)

                self._solver.solve(True)
                self._robot.update_kinematics()

                points_viz("holds", self._vis_holds)

                viz.display(self._robot.state.q)

            time.sleep(max(0, part_duration - (time.time() - t0)))
        print(f"End climb: {time.time() - t0}")
        return {name: self._robot.get_joint(name) for name in JOINTS}