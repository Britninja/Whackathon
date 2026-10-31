"""Géométrie du panneau : dimensions, repère en mm, numéro de trou d'une prise, vue de face.

Repère du panneau : origine en bas à gauche de la zone percée, x vers la droite, y vers le haut, en mm.
"""
import cv2
import numpy as np

HOLE_SPACING_MM = 15.0                       # distance entre deux trous voisins
N_HOLE_COLS, N_HOLE_ROWS = 52, 93
# Zone percée : une case de 15 mm par trou
BOARD_W_MM, BOARD_H_MM = N_HOLE_COLS * HOLE_SPACING_MM, N_HOLE_ROWS * HOLE_SPACING_MM      # 780 x 1395 mm
# Coin bas-gauche de la zone percée dans le repère du simulateur (le trou (0,0) y est en (17.5, 10) mm)
BOARD_ORIGIN_X_MM, BOARD_ORIGIN_Y_MM = 10.0, 2.5
# Coins de la zone percée en mm, dans l'ordre haut-gauche, haut-droite, bas-droite, bas-gauche
BOARD_CORNERS_MM = np.array([[0, BOARD_H_MM], [BOARD_W_MM, BOARD_H_MM], [BOARD_W_MM, 0], [0, 0]], np.float32)

# Décalage (mm) entre le centre de la tache de couleur d'une prise et le milieu de ses deux boulons.
# Valeur par défaut = simulateur ; sur le vrai mur chaque couleur a la sienne ("offset_mm" dans colors.py).
SIM_HOLD_OFFSET_MM = (-0.3, -13.0)

# Résolution de la vue de face
PX_PER_MM = 0.8
FRONT_W_PX, FRONT_H_PX = int(round(BOARD_W_MM * PX_PER_MM)), int(round(BOARD_H_MM * PX_PER_MM))


def mm_to_hole_index(x_mm, y_mm, offset_mm=None):
    """Centre d'une prise (mm) -> (colonne, ligne) de son trou de gauche. La prise est fixée par deux boulons ; le
    milieu des boulons est en (col + 1.5, row + 0.5) écarts, et la tache de couleur en est décalée de offset_mm."""
    dx, dy = offset_mm if offset_mm is not None else SIM_HOLD_OFFSET_MM
    return (int(round((x_mm - dx) / HOLE_SPACING_MM - 1.5)),
            int(round((y_mm - dy) / HOLE_SPACING_MM - 0.5)))


def board_corners_px(H_px2mm):
    """Les 4 coins de la zone percée en pixels (HG, HD, BD, BG), d'après une homographie pixel -> mm."""
    return cv2.perspectiveTransform(BOARD_CORNERS_MM.reshape(-1, 1, 2), np.linalg.inv(H_px2mm)).reshape(-1, 2)


def hole_spacing_px(H_px2mm):
    """Ecart entre deux trous, en pixels, mesuré au centre du panneau (moyenne horizontal / vertical)."""
    H_inv = np.linalg.inv(H_px2mm)

    def mm_to_px(x_mm, y_mm):
        return cv2.perspectiveTransform(np.float32([[[x_mm, y_mm]]]), H_inv)[0, 0]

    center = mm_to_px(BOARD_W_MM / 2, BOARD_H_MM / 2)
    return float(np.mean([np.hypot(*(mm_to_px(BOARD_W_MM / 2 + dx, BOARD_H_MM / 2 + dy) - center))
                          for dx, dy in ((HOLE_SPACING_MM, 0), (0, HOLE_SPACING_MM))]))


def make_front_view(image, H_px2mm):
    """Image du panneau vu de face (PX_PER_MM pixels par mm, y vers le bas comme toute image).
    Retourne (vue de face, homographie pixel caméra -> pixel vue de face)."""
    mm_to_front = np.array([[PX_PER_MM, 0, 0], [0, -PX_PER_MM, BOARD_H_MM * PX_PER_MM], [0, 0, 1]], np.float64)
    H_px2front = mm_to_front @ H_px2mm
    front = cv2.warpPerspective(image, H_px2front, (FRONT_W_PX, FRONT_H_PX), flags=cv2.INTER_AREA)
    return front, H_px2front
