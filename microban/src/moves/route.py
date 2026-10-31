"""
Mur d'escalade : lecture du JSON de perception, repère du mur, planning linéaire.

Repère du MUR (W), celui dans lequel on raisonne partout :
    x : gauche / droite sur le mur  (sens des colonnes croissantes)
    y : avance / recul              (+y = on s'éloigne du mur)
    z : vers le haut                (sens des lignes croissantes)

Le monde Placo n'a pas forcément les mêmes axes (dans le modèle du robot, le latéral
est souvent y et l'approche x). Au lieu de deviner, WallFrame calibre la
transformation W -> monde Placo au départ, à partir de la position réelle des 4 membres
posés sur les prises de départ du JSON. Plus besoin d'inverser x/y à la main.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LIMBS = ("right_foot", "left_foot", "right_hand", "left_hand")  # ordre d'un cycle
FEET = ("right_foot", "left_foot")
HANDS = ("right_hand", "left_hand")
LIMB_FRAMES = {
    "right_foot": "right_foot_ball",
    "left_foot": "left_foot_ball",
    "right_hand": "right_hand_ball",
    "left_hand": "left_hand_ball",
}

# Pas de la grille (col, row) en mètres.
# 15 mm donne 300 mm entre pieds et mains au départ (comme ta pose de départ actuelle)
# et 75 mm entre deux barreaux. À VÉRIFIER : la calibration affiche le pas estimé.
GRID_PITCH_M = 0.015

_EPS = 1e-9


@dataclass(frozen=True)
class Hold:
    col: int
    row: int


class Wall:
    """Prises d'un niveau, en coordonnées grille, converties en mètres dans le repère W."""

    def __init__(self, holds, start, pitch=GRID_PITCH_M, name=""):
        self.name = name
        self.pitch = float(pitch)
        # dict.fromkeys : supprime les doublons en gardant l'ordre
        self.holds = list(dict.fromkeys(Hold(int(h["col"]), int(h["row"])) for h in holds))
        self.start = {limb: Hold(int(start[limb]["col"]), int(start[limb]["row"])) for limb in LIMBS}

    @classmethod
    def from_json(cls, path, pitch=GRID_PITCH_M):
        data = json.loads(Path(path).read_text())
        return cls(data["holds"], data["start"], pitch=pitch, name=data.get("name", ""))

    def to_wall(self, hold):
        """Prise -> point (m) dans le repère W (le mur est le plan y = 0)."""
        return np.array([hold.col * self.pitch, 0.0, hold.row * self.pitch])

    def top_row(self):
        return max(h.row for h in self.holds)


def _kabsch(P, Q):
    """Rotation R et translation t qui minimisent ||R p + t - q|| (R propre, det = +1)."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


class WallFrame:
    """Transformation repère du mur W -> monde Placo."""

    def __init__(self, wall, R, t, away):
        self.wall = wall
        self.R = R          # colonnes = axes de W exprimés dans le monde Placo
        self.t = t
        self.away = away    # direction "recul" (on s'éloigne du mur) dans le monde Placo

    @classmethod
    def calibrate(cls, wall, limb_pos, body_pos):
        """
        limb_pos : {membre: position monde Placo} avec le robot posé sur wall.start
        body_pos : position monde du tronc (sert à savoir de quel côté du mur il est)
        Renvoie (frame, texte de diagnostic).
        """
        P = np.array([wall.to_wall(wall.start[l]) for l in LIMBS])
        Q = np.array([np.asarray(limb_pos[l], dtype=float) for l in LIMBS])
        R, t = _kabsch(P, Q)

        normal = R[:, 1]
        away = normal if np.dot(np.asarray(body_pos) - Q.mean(axis=0), normal) > 0 else -normal

        resid = np.linalg.norm(P @ R.T + t - Q, axis=1)
        ratios = [np.linalg.norm(Q[i] - Q[j]) / np.linalg.norm(P[i] - P[j])
                  for i in range(4) for j in range(i + 1, 4)
                  if np.linalg.norm(P[i] - P[j]) > _EPS]
        info = (f"[WallFrame] résidu max {1000 * resid.max():.1f} mm | "
                f"pas estimé {1000 * wall.pitch * np.mean(ratios):.1f} mm "
                f"(configuré {1000 * wall.pitch:.1f} mm) | "
                f"x_mur={np.round(R[:, 0], 2)} recul={np.round(away, 2)} z_mur={np.round(R[:, 2], 2)}")
        frame = cls(wall, R, t, away)
        frame.max_residual = float(resid.max())   # m : grand = robot pas sur les prises de départ
        return frame, info

    def hold_world(self, hold):
        return self.R @ self.wall.to_wall(hold) + self.t

    def delta_world(self, dx, dy, dz):
        """Déplacement exprimé dans W (x latéral, y recul, z haut) -> monde Placo."""
        return self.R[:, 0] * dx + self.away * dy + self.R[:, 2] * dz


def _next_hold(wall, state, limb, max_step_m, col_tol_m, min_gap_m, max_gap_m):
    """Prise la plus haute atteignable en ligne droite pour ce membre, ou None."""
    p = wall.pitch
    cur = state[limb]
    occupied = {h for l, h in state.items() if l != limb}
    best, best_key = None, None

    for h in wall.holds:
        if h in occupied or h.row <= cur.row:
            continue
        if (h.row - cur.row) * p > max_step_m + _EPS:      # trop loin pour un pas
            continue
        if abs(h.col - cur.col) * p > col_tol_m + _EPS:    # pas dans la "colonne" du membre
            continue
        if limb in FEET:
            # un pied reste au moins min_gap sous la main la plus basse
            low_hand = min(state[x].row for x in HANDS)
            if (low_hand - h.row) * p < min_gap_m - _EPS:
                continue
        else:
            # une main ne s'éloigne pas de plus de max_gap du pied le plus haut
            high_foot = max(state[x].row for x in FEET)
            if (h.row - high_foot) * p > max_gap_m + _EPS:
                continue

        key = (h.row, -abs(h.col - cur.col))
        if best_key is None or key > best_key:
            best, best_key = h, key
    return best


def plan_climb(wall, max_step_m=0.075, col_tol_m=0.02, min_gap_m=0.15, max_gap_m=0.30,
               order=LIMBS, max_moves=500):
    """
    Planning générique simple : cycles pied droit, pied gauche, main droite, main gauche ;
    chaque membre monte sur la prise la plus haute qui respecte les contraintes.
    S'arrête quand plus aucun membre ne peut monter. Renvoie [(membre, Hold), ...].
    """
    state = dict(wall.start)
    plan = []
    while len(plan) < max_moves:
        moved = False
        for limb in order:
            h = _next_hold(wall, state, limb, max_step_m, col_tol_m, min_gap_m, max_gap_m)
            if h is not None:
                plan.append((limb, h))
                state[limb] = h
                moved = True
        if not moved:
            break
    return plan