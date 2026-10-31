"""Etape 3, méthode de secours : utilisée seulement si hole_grid.fit_grid_homography échoue.

Elle part de l'homographie des 4 coins et arrondit chaque trou au nœud de grille le plus proche, en plusieurs passes à
tolérance décroissante. Elle a besoin d'une homographie de départ déjà assez bonne (à moins d'un demi-trou près).
Si tu constates qu'elle ne sert jamais sur tes images, ce fichier peut être supprimé (et son appel dans perception_rgb).
"""
import cv2
import numpy as np

from perception.geometry import BOARD_H_MM, BOARD_W_MM, HOLE_SPACING_MM, N_HOLE_COLS, N_HOLE_ROWS

CORNER_EXCLUSION_MM = 150   # sur le simulateur, les plaques ArUco cachent des trous et leurs carrés noirs ressemblent à des trous


def find_holes_inside_board(image, mask, px_per_mm, rough_H):
    """Centres des trous à l'intérieur du masque du panneau, en pixels, coins de 150 mm exclus."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    kernel_size = max(int(round(8 * px_per_mm)) | 1, 5)
    darkness = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    interior = cv2.erode(mask, np.ones((int(2 * px_per_mm) | 1,) * 2, np.uint8))
    H_inv = np.linalg.inv(rough_H)
    c = CORNER_EXCLUSION_MM
    for cx in (0.0, BOARD_W_MM):
        for cy in (0.0, BOARD_H_MM):
            sx, sy = (1 if cx == 0 else -1), (1 if cy == 0 else -1)
            square = np.float32([[cx, cy], [cx + sx * c, cy], [cx + sx * c, cy + sy * c], [cx, cy + sy * c]])
            square_px = cv2.perspectiveTransform(square.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
            cv2.fillConvexPoly(interior, square_px.astype(np.int32), 0)
    interior = interior > 0
    values = darkness[interior]
    if values.size == 0:
        return np.zeros((0, 2), np.float32)
    threshold = max(6.0, 0.5 * float(np.percentile(values, 90)))
    _, _, stats, centers = cv2.connectedComponentsWithStats(((darkness > threshold) & interior).astype(np.uint8))
    area_min, area_max = 0.5 * (1.6 * px_per_mm) ** 2 * np.pi, 4.0 * (3.2 * px_per_mm) ** 2
    keep = [i for i in range(1, len(stats)) if max(area_min, 2) <= stats[i, cv2.CC_STAT_AREA] <= area_max]
    return centers[keep].astype(np.float32)


def mean_fractional_shift(fractions):
    """Décalage moyen (dans [-0.5, 0.5)) d'une série de fractions d'écart, sans souci de repliement (moyenne circulaire)."""
    angles = 2 * np.pi * np.asarray(fractions)
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()) / (2 * np.pi))


def best_window_shift(counts, size, margin=4, max_shift=3):
    """Décalage (de -max_shift à +max_shift cases, 0 en priorité) de la fenêtre de `size` colonnes (ou lignes) qui
    contient le plus de trous, en pénalisant les colonnes peuplées juste à l'extérieur.
    counts[k] = nombre de trous de la colonne (ou ligne) k - margin."""
    best, best_score = 0, -1e18
    for shift in [0] + [d for n in range(1, max_shift + 1) for d in (-n, n)]:
        k = shift + margin
        if k < 1 or k + size >= len(counts):
            continue
        score = counts[k:k + size].sum() - counts[k - 1] - counts[k + size]
        if score > best_score:
            best, best_score = shift, score
    return best


def fit_grid_homography_fallback(image, mask, rough_H, px_per_mm):
    """Retourne (H pixel -> mm, trous px, trous mm, nombre de trous) ou None."""
    holes = find_holes_inside_board(image, mask, px_per_mm, rough_H)
    if len(holes) < 200:
        return None
    # on écarte aussi les trous dont la position en mm tombe dans un coin de 150 mm
    mm = cv2.perspectiveTransform(holes.reshape(-1, 1, 2), rough_H).reshape(-1, 2)
    near_side_x = (mm[:, 0] < CORNER_EXCLUSION_MM) | (mm[:, 0] > BOARD_W_MM - CORNER_EXCLUSION_MM)
    near_side_y = (mm[:, 1] < CORNER_EXCLUSION_MM) | (mm[:, 1] > BOARD_H_MM - CORNER_EXCLUSION_MM)
    holes = holes[~(near_side_x & near_side_y)]
    if len(holes) < 200:
        return None
    H = rough_H.copy()
    margin = 4
    holes_in = holes_mm_in = None
    for iteration, tolerance_mm in enumerate((6.0, 4.0, 2.5, 1.5)):
        mm = cv2.perspectiveTransform(holes.reshape(-1, 1, 2), H).reshape(-1, 2)
        u, v = mm[:, 0] / HOLE_SPACING_MM - 0.5, mm[:, 1] / HOLE_SPACING_MM - 0.5      # position en "numéro de trou"
        if iteration == 0:                       # recentrage global (la grille peut être décalée d'une fraction de trou)
            u = u - mean_fractional_shift(u - np.round(u))
            v = v - mean_fractional_shift(v - np.round(v))
        col, row = np.round(u), np.round(v)
        error_mm = np.hypot(u - col, v - row) * HOLE_SPACING_MM
        ok = (error_mm < tolerance_mm) & (col >= -margin) & (col < N_HOLE_COLS + margin) & (row >= -margin) & (row < N_HOLE_ROWS + margin)
        if ok.sum() < 150:
            return None
        if iteration == 1:                       # numéros absolus : fenêtre de 52 x 93 qui contient le plus de trous
            per_col = np.bincount((col[ok] + margin).astype(int), minlength=N_HOLE_COLS + 2 * margin + 1)
            per_row = np.bincount((row[ok] + margin).astype(int), minlength=N_HOLE_ROWS + 2 * margin + 1)
            col = col - best_window_shift(per_col, N_HOLE_COLS, margin)
            row = row - best_window_shift(per_row, N_HOLE_ROWS, margin)
        if iteration >= 1:
            ok &= (col >= 0) & (col < N_HOLE_COLS) & (row >= 0) & (row < N_HOLE_ROWS)
            if ok.sum() < 150:
                return None
        holes_mm = np.stack([(col + 0.5) * HOLE_SPACING_MM, (row + 0.5) * HOLE_SPACING_MM], axis=1)[ok]
        H_new, inliers = cv2.findHomography(holes[ok], holes_mm.astype(np.float32), cv2.RANSAC, max(1.0, tolerance_mm / 3))
        if H_new is None:
            return None
        H = H_new
        inliers = inliers.ravel().astype(bool)
        holes_in, holes_mm_in = holes[ok][inliers], holes_mm[inliers]
    return H, holes_in, holes_mm_in, int(len(holes_in))
