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

# TODO: Are these world or robot positions
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
        T_left_foot = np.eye(4)  # set left foot to
        # robot_frame_viz
        self._robot.set_T_world_frame("left_foot", T_left_foot)

        for name in self._robot.joint_names():
            print(name)
            print(self._robot.get_joint(name))
        # for limb in ["left_foot", "right_foot", "left_hand"]:
        #     print(f"{limb}:\t{self._robot.get_joint(limb)}")

        limb = "left_foot"  # (au debut gauche sets to origin)

        # floating base 6 degrees of freedom (Rotation + translation + 6dof)
        # left_foot is dans le model urdf
        # corps avec un repere fixe => urdf
        # ==> mujoco introduces sites => ce sont des frames?
        # T_world_left_foot =  self._robot.get_T_world_frame('left_foot')

        # simul ball position
        # sockets = hold centers

        # add_position_task(frame_name: str, target_world: ndarray) → DynamicsPositionTask
        # Adds a position (in the world) task.

        # tf from placo_utils
        # frame task = position + orientation
        # position trask
        # position = vector at 0,0,0 not eye (becase)
        # solver add_regularization_task (trouver la solution ou le robot bouge le petit possible)
        # jouer avec les pendaration pour l'importances des constraints
        # task.configure
        # TODO: what is the difference between hard and soft tasks constraint
        # mettre tache pour les boules => position not frame tasks
        # boule nach unten zeigend z.b. tache d'orientation (damit man den ball nicht anders hält)
        # masquees des axes quand j'utilise les taches
        # task.mask.set_axis("yz")
        # kinematics
        # weight: position % orientations
        # solver dump_task => debug
        # mettre des regularization pour le corps

        # meshcat (hat schon support von placo), ansonsten kann ich auch viser probieren

        # TODO: Where can I see the model info (xml sites, joints)
        # Parameters:
        # target_world (numpy.ndarray) – target position in the world
        holds = {"left_foot_ball": _START_HOLDS["left_foot_ball"]}
        task = self._solver.add_position_task(limb, holds[limb])
        task.configure(limb, "hard")

        # from leftPoint to world
        T_world_left = self._robot.get_T_world_frame("left_foot_ball")
        # robot = state and also a model

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
