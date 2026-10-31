"""Etape 3 : numéroter les trous pour obtenir une homographie précise pixel <-> mm.

Chaque trou visible reçoit son numéro (colonne, ligne), donc sa position réelle en mm. Avec des milliers de
correspondances pixel <-> mm, l'homographie est bien plus précise qu'avec 4 coins ou 4 marqueurs.

Deux façons de numéroter, selon la qualité de l'homographie de départ :
  fit_grid_homography                 départ approximatif (4 coins de l'étape 2) : on avance de voisin en voisin
  fit_grid_homography_from_markers    départ déjà juste à quelques mm (marqueurs ArUco) : chaque trou est
                                      simplement accroché au nœud de grille le plus proche
Les deux renvoient (H pixel -> mm, trous en pixels, mêmes trous en mm, nombre de trous) ou None.
"""
import cv2
import numpy as np

from perception.board_region import MAX_IMAGE_SIDE_PX, find_dark_spots, keep_grid_holes
from perception.geometry import (BOARD_H_MM, BOARD_W_MM, HOLE_SPACING_MM, N_HOLE_COLS, N_HOLE_ROWS,
                                 hole_spacing_px)

NOT_NUMBERED = -99999       # numéro d'un trou qu'on n'a pas réussi à numéroter


# ====================================================================== départ approximatif (sans marqueur)

def number_grid_holes(holes, rough_H, tolerance=0.3, max_jump=3):
    """Donne à chaque trou son numéro (colonne, ligne), relatif à un trou de départ près du centre.

    On part d'un trou bien entouré près du centre et on avance de voisin en voisin, comme on suit les lignes d'un
    quadrillage. Les vecteurs "un trou à droite" et "un trou en haut" sont mis à jour à chaque pas, ce qui suit la
    perspective et la distorsion de l'objectif sans avoir besoin d'une homographie parfaite. Un trou caché par une
    prise est franchi en sautant 2 ou 3 cases.
    Retourne un tableau (n, 2) d'entiers ; NOT_NUMBERED pour un trou non atteint."""
    n = len(holes)
    H_inv = np.linalg.inv(rough_H)

    def mm_to_px(x_mm, y_mm):
        return cv2.perspectiveTransform(np.float32([[[x_mm, y_mm]]]), H_inv)[0, 0]

    center_px = mm_to_px(BOARD_W_MM / 2, BOARD_H_MM / 2)
    spacing_px = hole_spacing_px(rough_H)
    # trou de départ : entouré de ses 4 voisins (pas isolé au milieu des prises), le plus proche du centre
    n_neighbors = np.zeros(n, np.int32)
    for start in range(0, n, 800):
        dist = np.hypot(holes[start:start + 800, None, 0] - holes[None, :, 0],
                        holes[start:start + 800, None, 1] - holes[None, :, 1])
        n_neighbors[start:start + 800] = ((dist >= 0.75 * spacing_px) & (dist <= 1.3 * spacing_px)).sum(axis=1)
    candidates = np.nonzero(n_neighbors >= 4)[0]
    if len(candidates) == 0:
        candidates = np.nonzero(n_neighbors >= 3)[0]
    if len(candidates) == 0:
        return np.full((n, 2), NOT_NUMBERED, np.int64)
    seed = int(candidates[np.argmin(np.hypot(holes[candidates, 0] - center_px[0], holes[candidates, 1] - center_px[1]))])
    seed_px = holes[seed]
    seed_mm = cv2.perspectiveTransform(np.float32([[seed_px]]), rough_H)[0, 0]
    step_right_0 = mm_to_px(*(seed_mm + [HOLE_SPACING_MM, 0])) - seed_px
    step_up_0 = mm_to_px(*(seed_mm + [0, HOLE_SPACING_MM])) - seed_px

    # index spatial (cases de la taille d'un écart entre trous) pour trouver vite le trou le plus proche d'un point
    cell_size = max(4.0, float(np.hypot(*step_right_0)))
    buckets = {}
    for k, (x, y) in enumerate(holes):
        buckets.setdefault((int(x // cell_size), int(y // cell_size)), []).append(k)

    def nearest_hole(target, radius):
        cx, cy = int(target[0] // cell_size), int(target[1] // cell_size)
        best, best_dist = -1, radius
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in buckets.get((cx + dx, cy + dy), ()):
                    dist = float(np.hypot(holes[k, 0] - target[0], holes[k, 1] - target[1]))
                    if dist < best_dist:
                        best, best_dist = k, dist
        return best

    numbers = np.full((n, 2), NOT_NUMBERED, np.int64)
    step_right, step_up = np.zeros((n, 2)), np.zeros((n, 2))       # vecteurs locaux, propres à chaque trou
    numbers[seed] = (0, 0)
    step_right[seed], step_up[seed] = step_right_0, step_up_0
    to_visit, hole_at = [seed], {(0, 0): seed}
    while to_visit:
        k = to_visit.pop()
        col, row = numbers[k]
        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):      # droite, gauche, haut, bas
            step_vec = (step_right[k] if axis == 0 else step_up[k]) * sign
            for jump in range(1, max_jump + 1):
                dcol, drow = (sign * jump, 0) if axis == 0 else (0, sign * jump)
                key = (int(col + dcol), int(row + drow))
                if key in hole_at:
                    break
                q = nearest_hole(holes[k] + step_vec * jump, tolerance * float(np.hypot(*step_vec)))
                if q < 0 or numbers[q, 0] != NOT_NUMBERED:
                    continue
                numbers[q] = key
                hole_at[key] = q
                measured_step = (holes[q] - holes[k]) / (jump * sign)
                step_right[q] = 0.6 * step_right[k] + 0.4 * measured_step if axis == 0 else step_right[k]
                step_up[q] = 0.6 * step_up[k] + 0.4 * measured_step if axis == 1 else step_up[k]
                to_visit.append(q)
                break
    return numbers


def best_full_window(counts, size):
    """Position de départ de la fenêtre de `size` colonnes (ou lignes) consécutives qui contient le plus de trous, en
    pénalisant les colonnes peuplées juste à l'extérieur. Sert à savoir quelle colonne numérotée est la colonne 0."""
    best, best_score = None, -1e18
    for start in range(0, len(counts) - size + 1) if len(counts) >= size else [0]:
        before = counts[start - 1] if start > 0 else 0
        after = counts[start + size] if start + size < len(counts) else 0
        score = counts[start:start + size].sum() - before - after
        if score > best_score:
            best, best_score = start, score
    return best


def fit_grid_homography(rough_H, holes):
    """Homographie pixel -> mm précise, à partir de la numérotation des trous (méthode principale sans marqueur).
    holes : trous de la grille en pixels (keep_grid_holes). Retourne (H, trous px, trous mm, nombre) ou None."""
    if len(holes) < 200:
        return None
    numbers = number_grid_holes(holes, rough_H)
    ok = numbers[:, 0] != NOT_NUMBERED
    if ok.sum() < 200:
        return None
    cols, rows = numbers[ok, 0], numbers[ok, 1]
    holes_per_col = np.bincount(cols - cols.min())
    holes_per_row = np.bincount(rows - rows.min())
    if len(holes_per_col) < N_HOLE_COLS - 1 or len(holes_per_row) < N_HOLE_ROWS - 1:
        return None
    # numéros relatifs -> numéros absolus (colonne 0 = bord gauche, ligne 0 = bas)
    first_col = cols.min() + best_full_window(holes_per_col, N_HOLE_COLS)
    first_row = rows.min() + best_full_window(holes_per_row, N_HOLE_ROWS)
    on_board = (cols >= first_col) & (cols < first_col + N_HOLE_COLS) & (rows >= first_row) & (rows < first_row + N_HOLE_ROWS)
    holes_px = holes[ok][on_board]
    holes_mm = np.stack([(cols[on_board] - first_col + 0.5) * HOLE_SPACING_MM,
                         (rows[on_board] - first_row + 0.5) * HOLE_SPACING_MM], 1).astype(np.float32)
    if len(holes_px) < 200:
        return None
    H, _ = cv2.findHomography(holes_px, holes_mm, cv2.RANSAC, 2.5)
    if H is None:
        return None
    for max_error_mm in (2.0, 1.5):            # on resserre : on ne garde que les trous bien placés et on recalcule
        error = np.hypot(*(cv2.perspectiveTransform(holes_px.reshape(-1, 1, 2), H).reshape(-1, 2) - holes_mm).T)
        good = error < max_error_mm
        if good.sum() < 150:
            return None
        H, _ = cv2.findHomography(holes_px[good], holes_mm[good], 0)
    error = np.hypot(*(cv2.perspectiveTransform(holes_px.reshape(-1, 1, 2), H).reshape(-1, 2) - holes_mm).T)
    good = error < 2.0
    return H, holes_px[good], holes_mm[good], int(good.sum())


# ====================================================================== départ déjà juste (marqueurs ArUco)

def snap_holes_to_nodes(holes, H, tolerance_mm):
    """Accroche chaque trou au nœud de grille le plus proche (un seul trou par nœud : le plus proche).
    Retourne (trous en pixels, nœuds en mm) des trous accrochés.

    C'est tout l'intérêt des marqueurs : l'homographie de départ étant juste à quelques mm, on sait d'emblée à quel
    trou (col, row) correspond chaque tache, sans numéroter de proche en proche."""
    mm = cv2.perspectiveTransform(holes.reshape(-1, 1, 2), H).reshape(-1, 2)
    col_row = np.round(mm / HOLE_SPACING_MM - 0.5)
    nodes = (col_row + 0.5) * HOLE_SPACING_MM
    error = np.hypot(*(mm - nodes).T)
    ok = ((error < tolerance_mm) & (col_row[:, 0] >= 0) & (col_row[:, 0] < N_HOLE_COLS)
          & (col_row[:, 1] >= 0) & (col_row[:, 1] < N_HOLE_ROWS))
    holes, nodes, error, col_row = holes[ok], nodes[ok], error[ok], col_row[ok]
    if len(holes) < 50:
        return holes, nodes
    node_id = col_row[:, 1] * N_HOLE_COLS + col_row[:, 0]
    order = np.lexsort((error, node_id))                          # par nœud, le plus proche d'abord
    first_of_node = np.ones(len(order), bool)
    first_of_node[1:] = node_id[order][1:] != node_id[order][:-1]
    selected = order[first_of_node]
    return holes[selected], nodes[selected].astype(np.float32)


def fit_grid_homography_from_markers(image, H_markers, margin_mm=20.0, min_holes=300):
    """Affine l'homographie des marqueurs avec la grille de trous : les marqueurs donnent 16 points aux coins, les
    trous en donnent des milliers répartis sur tout le panneau. Retourne (H, trous px, trous mm, nombre) ou None.

    L'homographie de départ étant déjà fiable, on jette d'emblée tout ce qui tombe hors du panneau (panneaux voisins,
    salle), on connaît l'écart attendu entre trous, et on accroche chaque trou à son nœud."""
    scale, biggest_side = 1.0, max(image.shape[:2])
    if biggest_side > 1.15 * MAX_IMAGE_SIDE_PX:          # find_dark_spots est réglé pour un trou tous les ~6 à 15 px
        scale = MAX_IMAGE_SIDE_PX / biggest_side
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image
    H_small = H_markers @ np.diag([1 / scale, 1 / scale, 1.0])          # pixels de l'image réduite -> mm
    holes = find_dark_spots(small)
    if len(holes) < min_holes:
        return None
    mm = cv2.perspectiveTransform(holes.reshape(-1, 1, 2), H_small).reshape(-1, 2)
    on_board = ((mm[:, 0] > -margin_mm) & (mm[:, 0] < BOARD_W_MM + margin_mm)
                & (mm[:, 1] > -margin_mm) & (mm[:, 1] < BOARD_H_MM + margin_mm))
    holes, _ = keep_grid_holes(holes[on_board], hole_spacing_px(H_small))
    if len(holes) < min_holes:
        return None
    H = H_small
    for tolerance in (0.35 * HOLE_SPACING_MM, 0.25 * HOLE_SPACING_MM, 0.15 * HOLE_SPACING_MM):   # on resserre
        holes_px, nodes_mm = snap_holes_to_nodes(holes, H, tolerance)
        if len(holes_px) < min_holes:
            return None
        H_new, _ = cv2.findHomography(holes_px, nodes_mm, cv2.RANSAC, 2.0)
        if H_new is None or not np.isfinite(H_new).all():
            return None
        H = H_new
    holes_px, nodes_mm = snap_holes_to_nodes(holes, H, 0.15 * HOLE_SPACING_MM)
    if len(holes_px) < min_holes:
        return None
    return H @ np.diag([scale, scale, 1.0]), holes_px / scale, nodes_mm, len(holes_px)   # ramenés à la pleine résolution
