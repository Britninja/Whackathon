from pathlib import Path

import microban_model
import numpy as np
import placo
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAP_PATH = PROJECT_ROOT / "simulator/climbing_sim/walls" / "N1.json"

with open(MAP_PATH, "r") as f:
    _HOLDS_MAP = json.load(f)

print(_HOLDS_MAP)

_START_HOLDS = {
    "right_foot": _HOLDS_MAP[0],
    "left_foot": _HOLDS_MAP[1],
    "right_hand": _HOLDS_MAP[2],
    "left_hand": _HOLDS_MAP[3],
}

_JOINTS = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
]


class PlacotestMove:

    def __init__(self) -> None:
        description = "Testing Placo"
        variants = ("classic",)  # its inverse kinematics uses the walking feet
        start_pose = "neutral"

        self._robot = placo.RobotWrapper(
            str(microban_model.URDF_DIR)
        )  # folder with robot.urdf
        # (frame names depend on your model)
        self._solver = placo.KinematicsSolver(self._robot)

        # Three limbs stay on their holds (hard constraints)...
        print("Start angles")
        print("self._robot.joint_names()")

        left_hand = np.eye(4)
        T_left_hand = np.eye(4)
        # self._robot.set_T_world_frame("left_foot", self._T_left_foot)

        for name in self._robot.joint_names():
            print(name)
            print(self._robot.get_joint(name))
        # for limb in ["left_foot", "right_foot", "left_hand"]:
        #     print(f"{limb}:\t{self._robot.get_joint(limb)}")
        # task = self._solver.add_position_task(limb, holds[limb])
        # task.configure(limb, "hard")

        # ... the fourth one reaches for the next hold (soft)
        # reach = self._solver.add_position_task("right_hand", next_hold)
        # reach.configure("reach", "soft", 1.0)

        # for _ in range(100):
        #     self._solver.solve(True)          # one QP step, applied to the robot
        #     self._robot.update_kinematics()

    def start(self, obs) -> None:
        self.t0 = obs["time"]
        self.stop_t0 = None

    def step(self, obs):
        t = obs["time"] - self.t0
        for name in self._robot.joint_names():
            if (self._robot.get_joint(name)) != 0:
                print(name)
                print(self._robot.get_joint(name))
        return {name: self._robot.get_joint(name) for name in self._robot.joint_names()}
