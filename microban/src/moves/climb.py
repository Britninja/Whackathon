import numpy as np
import mujoco
from collections import deque
from microban_model import JOINTS
from movePlaco import (BasePlacoMove, MoveRightLeg, MoveLeftLeg,
                       MoveRightArm, MoveLeftArm)


class ClimbMove:
    """CONTRÔLE de l'escalade : exécute une file d'actions via les MovePlaco.

    Perception et planning sont EXTERNES et appellent add_action().
    Le plan temporaire (_temporary_plan) lit N1 : à supprimer quand le vrai
    planning sera branché.
    """

    description = "Escalade : exécute les actions d'un planificateur (Placo)."
    variants = ("climber",)
    start_pose = "current"

    options = {
        "steps": 1,
        "move_duration_s": 5.0,
        "solver_iters": 5,
        "pause_s": 1.5,
        "absolute_targets": True,
    }
    option_docs = {
        "steps": "Nombre de cycles (plan temporaire uniquement)",
        "move_duration_s": "Durée d'une action (s)",
        "solver_iters": "Itérations du solveur Placo",
        "pause_s": "Pause entre deux actions (s)",
        "absolute_targets": "True : viser la position absolue de la prise ; "
                            "False : déplacement relatif entre prises",
    }

    LIMBS = ("right_foot", "left_foot", "right_hand", "left_hand")

    MOVES = {
        "right_foot": MoveRightLeg,
        "left_foot": MoveLeftLeg,
        "right_hand": MoveRightArm,
        "left_hand": MoveLeftArm,
    }

    FRAME_OF = {
        "right_foot": "right_foot_ball",
        "left_foot": "left_foot_ball",
        "right_hand": "right_hand_ball",
        "left_hand": "left_hand_ball",
    }

    # Jambes d'abord, puis bras
    CYCLE = ("right_foot", "left_foot", "right_hand", "left_hand")

    def __init__(self):
        self.sim = None
        self.queue = deque()
        self.move = None
        self.pause_until = 0.0
        self.last_cmd = None
        self.finished = False

    # ========================================================
    # INTERFACE AVEC LE FUTUR PLANNING
    # ========================================================
    def add_action(self, limb, target_mj, prev_mj=None):
        """Ajoute une action à exécuter.

        limb      : 'right_foot' | 'left_foot' | 'right_hand' | 'left_hand'
        target_mj : position cible de la prise (m, repère monde MuJoCo)
        prev_mj   : position de la prise actuelle du membre (m, repère MuJoCo),
                    utilisée seulement si absolute_targets = False
        """
        t = np.asarray(target_mj, float)
        p = None if prev_mj is None else np.asarray(prev_mj, float)
        self.queue.append((limb, t, p))

    # ========================================================
    # PLAN TEMPORAIRE (à supprimer avec le vrai planning)
    # ========================================================
    def _temporary_plan(self, wall):
        cur = {l: wall.start[l] for l in self.LIMBS}
        for _ in range(int(self.options["steps"])):
            for limb in self.CYCLE:
                above = [h for h in wall.holds
                         if h.col == cur[limb].col and h.row > cur[limb].row]
                if not above:
                    print(f"[Climb] plus de prise au-dessus pour {limb}")
                    return
                tgt = min(above, key=lambda h: h.row)
                self.add_action(
                    limb,
                    target_mj=np.asarray(tgt.socket, float) + np.array([0.0, 0.0, 0.001]),
                    prev_mj=np.asarray(cur[limb].socket, float) + np.array([0.0, 0.0, 0.001]),
                )
                cur[limb] = tgt

    # ========================================================
    # DIAGNOSTIC AU DÉMARRAGE
    # ========================================================
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

    def _print_diagnostic(self, obs):
        probe = MoveRightLeg()
        probe.start(obs)  # synchronise Placo avec MuJoCo
        wall = self.sim.wall

        print("\n[DIAG] tronc MuJoCo :", np.round(self.sim.data.qpos[0:3], 4))
        for limb, frame in self.FRAME_OF.items():
            p_placo = probe.model.get_T_world_frame(frame)[:3, 3]
            p_mj = self._mj_pos(frame)
            sock = np.asarray(wall.start[limb].socket, float)
            print(f"[DIAG] {limb}")
            print(f"   boule Placo    = {np.round(p_placo, 4)}")
            print(f"   boule MuJoCo   = {None if p_mj is None else np.round(p_mj, 4)}")
            print(f"   prise (socket) = {np.round(sock, 4)}")
            print(f"   Placo - prise  = {np.round((p_placo - sock) * 1000, 1)} mm")

        # Axes du mur : quelle coordonnée change avec col, avec row ?
        h = {(x.col, x.row): np.asarray(x.socket, float) for x in wall.holds}
        if (26, 2) in h and (26, 7) in h and (21, 2) in h:
            print("[DIAG] +5 rows =", np.round(h[(26, 7)] - h[(26, 2)], 4))
            print("[DIAG] -5 cols =", np.round(h[(21, 2)] - h[(26, 2)], 4))
        print()

    # ========================================================
    # CONVERSION D'UN VECTEUR MuJoCo -> Placo (mode relatif)
    # ========================================================
    def _vec_mj_to_placo(self, v_mj):
        qpos = self.sim.data.qpos
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(qpos[3:7], float))
        R = R.reshape(3, 3)
        T = self.move.model.get_T_world_fbase()
        return T[:3, :3] @ (R.T @ np.asarray(v_mj, float))

    # ========================================================
    # DÉMARRAGE D'UNE ACTION
    # ========================================================
    def _start_action(self, obs):
        limb, target, prev = self.queue.popleft()

        self.move = self.MOVES[limb]()
        self.move.options = dict(self.move.options)  # options = dict de classe partagé
        self.move.options["duration_s"] = float(self.options["move_duration_s"])
        self.move.options["solver_iters"] = int(self.options["solver_iters"])
        self.move.start(obs)  # fixe p0 = position réelle de la boule

        if self.options["absolute_targets"]:
            delta = target - self.move.p0
        else:
            delta = self._vec_mj_to_placo(target - prev)

        d_cm = delta * 100.0
        (self.move.options["dx_cm"],
         self.move.options["dy_cm"],
         self.move.options["dz_cm"]) = d_cm

        print(f"[Climb] {limb} p0={np.round(self.move.p0, 3)} "
              f"delta={np.round(d_cm, 1)} cm")

    # ========================================================
    # API MOVE
    # ========================================================
    def start(self, obs):
        self.sim = self.robot.controller.sim
        BasePlacoMove.sim = self.sim  # pour la synchro du tronc dans movePlaco

        self.queue.clear()
        self.move = None
        self.finished = False
        self.pause_until = 0.0
        self.last_cmd = None

        self._print_diagnostic(obs)
        self._temporary_plan(self.sim.wall)  # <- à retirer avec le vrai planning
        print(f"[Climb] {len(self.queue)} actions en file.")

    def step(self, obs):
        # Action en cours
        if self.move is not None:
            cmd = self.move.step(obs)
            self.last_cmd = cmd

            # --- DEBUG : consigne Placo vs réalité MuJoCo (2 fois par seconde) ---
            self._n = getattr(self, "_n", 0) + 1
            if self._n % 25 == 0:   # à 50 Hz : deux fois par seconde
                f = self.move.FRAME
                tgt = self.move.tasks[f].target_world
                real = self._mj_pos(f)
                trunk = self.sim.data.qpos[0:3]
                print(f"[DBG] cible={np.round(tgt, 3)} réel={np.round(real, 3)} "
                      f"écart={np.round((tgt - real) * 1000, 0)} mm "
                      f"tronc={np.round(trunk, 3)}")
                for k in ("right_knee", "left_knee"):
                    i = JOINTS.index(k)
                    print(f"       {k}: commande={cmd[k]:.2f} réel={obs['q'][i]:.2f}")

            if obs["time"] - self.move.t0 >= self.move.options["duration_s"]:
                f = self.move.FRAME
                err = np.linalg.norm(
                    self.move.tasks[f].target_world - self.move._current_pos()
                )
                print(f"[Climb] fin {f} : erreur Placo = {err * 1000:.1f} mm")
                self.move = None
                self.pause_until = obs["time"] + float(self.options["pause_s"])
            return cmd

        # Pause entre deux actions
        if obs["time"] < self.pause_until:
            return self.last_cmd

        # File vide : on garde la dernière pose et on attend d'autres actions
        if not self.queue:
            if not self.finished:
                self.finished = True
                print("[Climb] File vide, en attente.")
            return self.last_cmd

        self.finished = False
        self._start_action(obs)
        cmd = self.move.step(obs)
        self.last_cmd = cmd
        return cmd