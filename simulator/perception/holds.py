"""Etape 4 : les prises, cherchées par leur couleur dans la vue de face du panneau.

Dans la vue de face, le fond de la salle n'existe plus et une prise a la même taille partout (en mm). Pour chaque
couleur de colors.py :
  - une prise isolée = le centre de sa tache ;
  - plusieurs prises collées = on apprend la forme d'une prise isolée sur l'image et on la pose sur les positions de
    grille possibles ;
  - s'il y a trop peu de prises de cette couleur pour apprendre leur forme (souvent le vert) : on coupe chaque tache
    en 1, 2, 3... parties (k-means) jusqu'à ce que chaque partie ait la taille d'une prise seule ;
  - le vert pâle ressemble au bois : on le sépare par la composante a* (Lab) et un écart minimal au bois voisin.

Fonction principale : detect_holds(front_view, colors).
"""
import cv2
import numpy as np

from perception import colors as color_settings      # lu à chaque appel : les réglages restent modifiables à chaud
from perception.geometry import (BOARD_H_MM, HOLE_SPACING_MM, N_HOLE_COLS, N_HOLE_ROWS, PX_PER_MM,
                                 SIM_HOLD_OFFSET_MM, mm_to_hole_index)

HOLD_MIN_AREA_MM2 = 300.0     # en dessous, une tache de couleur est du bruit, pas une prise

# Une prise SEULE tient toujours dans cette boîte (largeur, hauteur en mm) dans la vue de face. Mesuré sur une vraie
# photo : une prise seule fait au plus ~66 mm de large et ~73 mm de haut (on voit plus ou moins ses flancs selon
# l'angle) ; deux prises collées font au moins ~100 mm de large (côte à côte) ou ~81 mm de haut (l'une sur l'autre).
# La largeur max est prise sous le milieu (80 et non 83) : coupé en 2, un groupe de 3 prises donne des moitiés de ~89 mm.
# Sert seulement aux couleurs qui ont trop peu de prises pour apprendre leur forme (souvent le vert).
MAX_SINGLE_HOLD_MM = (80, 77)
_TEMPLATE_HALF = int(40 * PX_PER_MM)     # demi-taille du gabarit appris, en pixels de la vue de face


# ---------------------------------------------------------------------- séparer des prises collées

def kmeans(pixels, centers, n_iterations=10):
    """k-means sur des pixels (N, 2) à partir de centres de départ. Retourne (centres, n° de partie de chaque pixel)."""
    for _ in range(n_iterations):
        nearest = np.argmin(((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        centers = np.array([pixels[nearest == k].mean(axis=0) if (nearest == k).any() else centers[k]
                            for k in range(len(centers))])
    nearest = np.argmin(((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    return centers, nearest


def fits_in_one_hold(pixels):
    """Ces pixels tiennent-ils dans la boîte d'une prise seule (MAX_SINGLE_HOLD_MM) ? Mesuré d'un bord à l'autre,
    comme les mesures de référence (le masque a déjà été nettoyé de ses pixels isolés)."""
    width_mm, height_mm = (pixels.max(axis=0) - pixels.min(axis=0)) / PX_PER_MM
    return width_mm <= MAX_SINGLE_HOLD_MM[0] and height_mm <= MAX_SINGLE_HOLD_MM[1]


def split_blob_by_size(blob, max_holds=6):
    """Coupe une tache en 1, 2, 3... parties (k-means) jusqu'à ce que chaque partie tienne dans la boîte d'une prise
    seule. Retourne le centre (u, v) de chaque partie, en pixels de la vue de face. Une prise seule garde le centre de
    sa tache ; deux prises côte à côte sont coupées au milieu.
    Méthode de secours pour une couleur qui a trop peu de prises pour apprendre leur forme (souvent le vert)."""
    ys, xs = np.nonzero(blob)
    pixels = np.column_stack([xs, ys]).astype(np.float64)
    # départ du k-means : k points répartis le long de l'axe principal de la tache (déterministe)
    mean = pixels.mean(axis=0)
    axis = np.linalg.eigh(np.cov((pixels - mean).T))[1][:, -1]
    along = np.sort((pixels - mean) @ axis)
    for k in range(1, max_holds + 1):
        starts = np.array([mean + axis * along[int((i + 0.5) / k * (len(along) - 1))] for i in range(k)])
        centers, part = kmeans(pixels, starts)
        if all(fits_in_one_hold(pixels[part == i]) for i in range(k) if (part == i).sum() > 0):
            break
    return centers


def learn_hold_shape(mask):
    """Forme moyenne d'une prise isolée, apprise sur l'image (moyenne des taches de taille typique, centrées).
    Retourne (gabarit booléen, aire typique d'une prise en px, demi-taille du gabarit) ou None si trop peu de prises."""
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in range(1, n)])
    areas = areas[areas > HOLD_MIN_AREA_MM2 * PX_PER_MM * PX_PER_MM]
    if len(areas) < 3:
        return None
    typical_area = float(np.percentile(areas, 25))
    half = _TEMPLATE_HALF
    total, count = np.zeros((2 * half + 1, 2 * half + 1), np.float32), 0
    for i in range(1, n):
        if 0.8 * typical_area <= stats[i, cv2.CC_STAT_AREA] <= 1.25 * typical_area:
            cx, cy = centers[i]
            shift = np.float32([[1, 0, half - cx], [0, 1, half - cy]])
            total += cv2.warpAffine((labels == i).astype(np.float32), shift, (2 * half + 1, 2 * half + 1), flags=cv2.INTER_LINEAR)
            count += 1
    if count < 3:
        return None
    return (total / count) > 0.5, typical_area, half


def split_blob_with_template(blob, template, half, min_coverage=0.8, offset_mm=None):
    """Plusieurs prises collées dans une même tache : on pose le gabarit d'une prise sur chaque position de grille
    possible et on garde celles où il est presque entièrement dans la tache, sans que deux prises se chevauchent
    (deux prises ne peuvent pas être à moins de 2 colonnes ET 2 lignes l'une de l'autre, sauf en diagonale exacte).
    Retourne les centres (x, y) en mm."""
    dx, dy = offset_mm if offset_mm is not None else SIM_HOLD_OFFSET_MM
    tmpl_ys, tmpl_xs = np.nonzero(template)
    tmpl_area = float(template.sum())
    bx, by, bw, bh = cv2.boundingRect(blob.astype(np.uint8))
    candidates = []
    for col in range(0, N_HOLE_COLS - 2):
        for row in range(0, N_HOLE_ROWS):
            u = ((col + 1.5) * HOLE_SPACING_MM + dx) * PX_PER_MM
            v = (BOARD_H_MM - ((row + 0.5) * HOLE_SPACING_MM + dy)) * PX_PER_MM
            if not (bx - half <= u <= bx + bw + half and by - half <= v <= by + bh + half):
                continue
            ys, xs = tmpl_ys - half + int(round(v)), tmpl_xs - half + int(round(u))
            in_image = (ys >= 0) & (ys < blob.shape[0]) & (xs >= 0) & (xs < blob.shape[1])
            if in_image.sum() < 0.9 * tmpl_area:
                continue
            coverage = blob[ys[in_image], xs[in_image]].sum() / tmpl_area
            if coverage >= min_coverage:
                candidates.append((coverage, col, row))
    candidates.sort(key=lambda c: -c[0])
    kept = []
    for coverage, col, row in candidates:
        overlaps = any(abs(col - c2) <= 2 and abs(row - r2) <= 2 and not (abs(col - c2) == 2 and abs(row - r2) == 2)
                       for _, c2, r2 in kept)
        if not overlaps:
            kept.append((coverage, col, row))
    return [((col + 1.5) * HOLE_SPACING_MM + dx, (row + 0.5) * HOLE_SPACING_MM + dy) for _, col, row in kept]


# ---------------------------------------------------------------------- détection

def detect_holds(front, colors=None):
    """Prises dans la vue de face. Retourne une liste de dict :
    name, bgr (couleur de dessin), x, y (mm, repère du panneau), col, row (trou), u, v (pixels de la vue de face),
    offset_mm (décalage de la couleur, à repasser à mm_to_hole_index)."""
    blurred = cv2.GaussianBlur(front, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab_a = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.int16) - 128      # < 0 : vers le vert
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(4 * PX_PER_MM) | 1, int(4 * PX_PER_MM) | 1))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(5 * PX_PER_MM) | 1, int(5 * PX_PER_MM) | 1))
    area_min = HOLD_MIN_AREA_MM2 * PX_PER_MM * PX_PER_MM
    palette = color_settings.COLORS if colors is None else colors
    holds = []

    def add_hold(name, color, x_mm, y_mm, u_px, v_px, offset_mm):
        col, row = mm_to_hole_index(x_mm, y_mm, offset_mm)
        # offset_mm est gardé dans la prise : wall_holds / sim_perception en ont besoin pour recalculer (col, row)
        # après avoir fusionné les deux caméras
        holds.append({"name": name, "bgr": color["draw_bgr"], "u": u_px, "v": v_px,
                      "x": x_mm, "y": y_mm, "col": col, "row": row, "offset_mm": offset_mm})

    # Le liseré clair des prises franches (orange, bleu) ressemble à du vert pâle : on masque leurs alentours
    # avant de chercher le vert.
    solid_holds = None
    if any("lab_a_max" in c for c in palette.values()):
        solid_holds = np.zeros(front.shape[:2], np.uint8)
        for c in palette.values():
            if "lab_a_max" not in c:
                solid_holds |= cv2.inRange(hsv, c["hsv_min"], c["hsv_max"])
        solid_holds = cv2.dilate(cv2.morphologyEx(solid_holds, cv2.MORPH_OPEN, open_kernel),
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(10 * PX_PER_MM) | 1,) * 2))
    local_board_a = None            # a* du bois autour de chaque pixel (calculé seulement si besoin)

    for name, color in palette.items():
        offset_mm = color.get("offset_mm")
        mask = cv2.inRange(hsv, color["hsv_min"], color["hsv_max"])
        if "lab_a_max" in color:
            mask = mask & ((lab_a <= color["lab_a_max"]).astype(np.uint8) * 255)
            if solid_holds is not None:
                mask[solid_holds > 0] = 0
            if color_settings.GREEN_MIN_CONTRAST:
                if local_board_a is None:
                    local_board_a = cv2.medianBlur((lab_a + 128).astype(np.uint8), 101).astype(np.int16) - 128
                mask = mask & (((local_board_a - lab_a) >= color_settings.GREEN_MIN_CONTRAST).astype(np.uint8) * 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        shape = learn_hold_shape(mask)
        if shape is None:          # trop peu de prises de cette couleur pour apprendre leur forme : découpe par taille
            n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                if area <= area_min:                                # trop petit : bruit
                    continue
                for u, v in split_blob_by_size(labels == i):
                    add_hold(name, color, u / PX_PER_MM, BOARD_H_MM - v / PX_PER_MM, float(u), float(v), offset_mm)
            continue

        template, typical_area, half = shape
        n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 0.6 * typical_area:                           # trop petit : bruit
                continue
            if area <= 1.35 * typical_area:                         # une prise seule
                u, v = float(centers[i][0]), float(centers[i][1])
                add_hold(name, color, u / PX_PER_MM, BOARD_H_MM - v / PX_PER_MM, u, v, offset_mm)
            else:                                                   # plusieurs prises collées
                for x, y in split_blob_with_template(labels == i, template, half, offset_mm=offset_mm):
                    add_hold(name, color, x, y, x * PX_PER_MM, (BOARD_H_MM - y) * PX_PER_MM, offset_mm)
    return holds