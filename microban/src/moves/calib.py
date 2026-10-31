import numpy as np
import mujoco

try:
    from movePlaco import (BasePlacoMove, MoveRightLeg, MoveLeftLeg,
                           MoveRightArm, MoveLeftArm)
except ImportError:
    from moves.movePlaco import (BasePlacoMove, MoveRightLeg, MoveLeftLeg,
                                 MoveRightArm, MoveLeftArm)


class CalibMove:
    """Calibration : affiche les repères Placo / MuJoCo / prises, puis déplace
    UN membre et compare le déplacement voulu au déplacement réel."""

    description = "Calibration des frames Placo / MuJoCo."
    variants = ("climber",)
    start_pose = "current"

    options = {
        "limb": "right_foot",
        "dx_cm": 0.0,
        "dy_cm": 0.0,
        "dz_cm": 3.0,
        "duration_s": 4.0,
        "solver_iters": 5,
        "straight": False,
    }
    option_docs = {
        "limb": "right_foot | left_foot | right_hand | left_hand",
        "dx_cm": "déplacement X (cm)",
        "dy_cm": "déplacement Y (cm)",
        "dz_cm": "déplacement Z (cm)",
        "duration_s": "durée (s)",
        "solver_iters": "itérations",
        "straight": "True : ligne droite (sans LIFT ni BACK), pour isoler le repère",
    }

    MOVES = {
        "right_foot": MoveRightLeg,
        "left_foot": MoveLeftLeg,
        "right_hand": MoveRightArm,
        "left_hand": MoveLeftArm,
    }
    FRAMES = {
        "right_foot": "right_foot_ball",
        "left_foot": "left_foot_ball",
        "right_hand": "right_hand_ball",
        "left_hand": "left_hand_ball",
    }

    def __init__(self):
        self.sim = None
        self.move = None
        self.done = False
        self.last_cmd = None
        self.frame = None
        self.mj0 = None

    def _mj_pos(self, name):
        """Position monde MuJoCo d'un site / corps / géom nommé (None si introuvable)."""
        m, d = self.sim.model, self.sim.data
        for obj, arr in ((mujoco.mjtObj.mjOBJ_SITE, d.site_xpos),
                         (mujoco.mjtObj.mjOBJ_BODY, d.xpos),
                         (mujoco.mjtObj.mjOBJ_GEOM, d.geom_xpos)):
            try:
                i = mujoco.mj_name2id(m, obj, name)
            except Exception:
                i = -1
            if i >= 0:
                return arr[i].copy()
        return None

    def start(self, obs):
        self.sim = self.robot.controller.sim
        BasePlacoMove.sim = self.sim
        wall = self.sim.wall
        limb = self.options["limb"]

        self.move = self.MOVES[limb]()
        self.move.options = dict(self.move.options)
        self.move.options.update(
            dx_cm=float(self.options["dx_cm"]),
            dy_cm=float(self.options["dy_cm"]),
            dz_cm=float(self.options["dz_cm"]),
            duration_s=float(self.options["duration_s"]),
            solver_iters=int(self.options["solver_iters"]),
        )
        if self.options["straight"]:
            self.move.LIFT = 0.0
            self.move.BACK = 0.0
        self.move.start(obs)

        # Noms MuJoCo contenant "ball" (pour retrouver les bons si None plus bas)
        m = self.sim.model
        counts = {"site": m.nsite, "body": m.nbody, "geom": m.ngeom}
        kinds = {"site": mujoco.mjtObj.mjOBJ_SITE,
                 "body": mujoco.mjtObj.mjOBJ_BODY,
                 "geom": mujoco.mjtObj.mjOBJ_GEOM}
        for label in ("site", "body", "geom"):
            names = [mujoco.mj_id2name(m, kinds[label], i) for i in range(counts[label])]
            names = [n for n in names if n and "ball" in n.lower()]
            if names:
                print(f"[CALIB] {label}s MuJoCo avec 'ball' : {names}")

        # Positions de départ
        print("\n[CALIB] tronc MuJoCo :", np.round(self.sim.data.qpos[0:3], 4))
        for l, frame in self.FRAMES.items():
            p_placo = self.move.model.get_T_world_frame(frame)[:3, 3]
            p_mj = self._mj_pos(frame)
            sock = np.asarray(wall.start[l].socket, float)
            print(f"[CALIB] {l}")
            print(f"   Placo  = {np.round(p_placo, 4)}")
            print(f"   MuJoCo = {None if p_mj is None else np.round(p_mj, 4)}")
            print(f"   prise  = {np.round(sock, 4)}")
            print(f"   prise - Placo = {np.round((sock - p_placo) * 1000, 1)} mm")
            if p_mj is not None:
                print(f"   MuJoCo - Placo = {np.round((p_mj - p_placo) * 1000, 1)} mm")

        # Axes du mur : quelle coordonnée change avec col, avec row ?
        h = {(x.col, x.row): np.asarray(x.socket, float) for x in wall.holds}
        if (26, 2) in h and (26, 7) in h and (21, 2) in h:
            print("[CALIB] +5 rows =", np.round(h[(26, 7)] - h[(26, 2)], 4))
            print("[CALIB] -5 cols =", np.round(h[(21, 2)] - h[(26, 2)], 4))

        self.frame = self.FRAMES[limb]
        self.mj0 = self._mj_pos(self.frame)
        self.done = False
        print(f"\n[CALIB] test : {limb}, delta voulu = "
              f"({self.options['dx_cm']}, {self.options['dy_cm']}, "
              f"{self.options['dz_cm']}) cm\n")

    def step(self, obs):
        if self.done:
            return self.last_cmd

        cmd = self.move.step(obs)
        self.last_cmd = cmd

        # +1 s après la fin pour laisser MuJoCo rattraper la consigne
        if obs["time"] - self.move.t0 >= self.move.options["duration_s"] + 1.0:
            want = np.array([self.options["dx_cm"],
                             self.options["dy_cm"],
                             self.options["dz_cm"]], float)
            placo_moved = (self.move._current_pos() - self.move.p0) * 100.0
            print("[CALIB] RÉSULTAT")
            print(f"   voulu          = {np.round(want, 2)} cm")
            print(f"   Placo a bougé  = {np.round(placo_moved, 2)} cm")
            mj_now = self._mj_pos(self.frame)
            if mj_now is not None and self.mj0 is not None:
                print(f"   MuJoCo a bougé = {np.round((mj_now - self.mj0) * 100.0, 2)} cm")
            print(f"   tronc MuJoCo   = {np.round(self.sim.data.qpos[0:3], 4)}")
            self.done = True
        return cmd