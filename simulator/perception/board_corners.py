"""Etape 2 (sans marqueur) : les 4 coins du panneau, à partir de son masque.

On pose d'abord un rectangle grossier autour du contour, puis on recale chaque côté sur le vrai contour, puis au
sous-pixel là où la densité de trous s'arrête. Les coins sont les intersections des 4 côtés.

Fonction principale : fit_board_quad(contour, density, threshold). Puis les contrôles : quad_is_plausible,
sides_cut_by_image_border, corners_inside_image.
"""
import cv2
import numpy as np

from perception.geometry import BOARD_CORNERS_MM

BORDER_MARGIN_PX = 8        # un coin à moins de 8 px du bord de l'image n'est pas un vrai coin


# ---------------------------------------------------------------------- outils sur les droites

def order_corners(points):
    """Range 4 points dans l'ordre : haut-gauche, haut-droite, bas-droite, bas-gauche."""
    points = np.asarray(points, np.float32)
    by_y = points[np.argsort(points[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], np.float32)


def fit_line(points):
    """Droite robuste (Huber) passant par des points. Retourne (un point, direction)."""
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)


def corners_from_sides(side_lines, fallback):
    """Coin i = intersection du côté i-1 et du côté i. Si deux côtés sont parallèles, renvoie fallback."""
    corners = []
    for i in range(4):
        (point_a, dir_a), (point_b, dir_b) = side_lines[(i - 1) % 4], side_lines[i]
        system = np.array([dir_a, -dir_b]).T
        if abs(np.linalg.det(system)) < 1e-6:
            return fallback
        t = np.linalg.solve(system, point_b - point_a)
        corners.append(point_a + t[0] * dir_a)
    return np.array(corners, np.float32)


# ---------------------------------------------------------------------- du contour aux 4 coins

def rough_quad(contour):
    """Quadrilatère grossier : plus petit rectangle (éventuellement tourné) qui contient le contour."""
    hull = cv2.convexHull(contour)
    if len(hull) < 4:
        return None
    return order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))


def snap_quad_to_contour(contour, quad, band_px=90.0):
    """Recale chaque côté du quadrilatère sur les points du contour qui en sont proches (partie centrale du côté
    seulement, les coins sont souvent abîmés), puis recalcule les coins."""
    points = contour.reshape(-1, 2).astype(np.float64)
    side_lines = []
    for i in range(4):
        start, end = quad[i].astype(np.float64), quad[(i + 1) % 4].astype(np.float64)
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 20:
            return quad
        direction /= length
        normal = np.array([direction[1], -direction[0]])
        position_along = (points - start) @ direction / length       # 0 au début du côté, 1 à la fin
        distance_to_side = np.abs((points - start) @ normal)
        selected = points[(position_along > 0.2) & (position_along < 0.8) & (distance_to_side < band_px)]
        if len(selected) < 20:
            return quad
        side_lines.append(fit_line(selected))
    return corners_from_sides(side_lines, quad)


def snap_quad_to_density_edge(density, threshold, quad, half_window=35, n_samples=40):
    """Recalage fin (sous le pixel) : le long de la normale à chaque côté, on cherche où la densité de trous passe sous
    le seuil, on ajuste une droite sur ces points, et les coins sont les intersections des 4 droites."""
    center = quad.mean(axis=0)
    side_lines = []
    for i in range(4):
        start, end = quad[i], quad[(i + 1) % 4]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 20:
            return quad
        direction /= length
        normal = np.array([direction[1], -direction[0]], np.float32)
        if np.dot(normal, (start + end) / 2 - center) < 0:
            normal = -normal                                          # normale vers l'extérieur du panneau
        along = np.linspace(0.2, 0.8, n_samples, dtype=np.float32)    # on évite les coins
        offsets = np.arange(-half_window, half_window + 1, dtype=np.float32)
        base_points = start[None, :] + along[:, None] * (end - start)[None, :]
        sample_points = base_points[:, None, :] + offsets[None, :, None] * normal[None, None, :]
        profiles = cv2.remap(density, sample_points[..., 0].astype(np.float32), sample_points[..., 1].astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        edge_points = []
        for k in range(n_samples):
            profile = profiles[k]
            if not (profile[: half_window + 8] >= threshold).any():
                continue
            above = profile >= threshold
            exit_idx = None                       # dernier passage "dans le panneau" -> "dehors"
            for j in range(len(profile) - 1, 0, -1):
                if above[j - 1] and not above[j]:
                    exit_idx = j
                    break
            if exit_idx is None:
                continue
            inside_val, outside_val = profile[exit_idx - 1], profile[exit_idx]
            frac = (inside_val - threshold) / (inside_val - outside_val + 1e-6)     # interpolation linéaire
            edge_points.append(base_points[k] + (offsets[exit_idx - 1] + frac) * normal)
        if len(edge_points) < 10:
            return quad
        side_lines.append(fit_line(np.array(edge_points, np.float32)))
    return corners_from_sides(side_lines, quad)


def fit_board_quad(contour, density, threshold):
    """Les 4 coins du panneau (HG, HD, BD, BG) en pixels, ou None : rectangle grossier, recalé 2 fois sur le contour
    (bande de 90 px puis 40 px), puis au sous-pixel sur la carte de densité."""
    quad = rough_quad(contour)
    if quad is None:
        return None
    quad = snap_quad_to_contour(contour, quad)
    quad = snap_quad_to_contour(contour, quad, band_px=40.0)
    return snap_quad_to_density_edge(density, threshold, quad)


# ---------------------------------------------------------------------- contrôles

def quad_is_plausible(quad):
    """Convexe, angles entre 45° et 135°, et rapport largeur/hauteur proche de celui du panneau (~0.56)."""
    q = np.asarray(quad, np.float64)
    turn_signs = []
    for i in range(4):
        prev_pt, pt, next_pt = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = prev_pt - pt, next_pt - pt
        turn_signs.append(np.sign(v1[0] * v2[1] - v1[1] * v2[0]))
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        if not 45 < np.degrees(np.arccos(np.clip(cos_angle, -1, 1))) < 135:
            return False
    if len(set(turn_signs)) != 1:
        return False
    width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return 0.33 < width / height < 0.85          # une simple bande de prises donnerait ~0.24


def sides_cut_by_image_border(mask):
    """Côtés de l'image que le masque du panneau touche : le panneau y est coupé."""
    m = BORDER_MARGIN_PX
    bands = {"gauche": mask[:, :m], "droite": mask[:, -m:], "haut": mask[:m, :], "bas": mask[-m:, :]}
    return [side for side, band in bands.items() if np.count_nonzero(band) > 40]


def corners_inside_image(quad, image_shape):
    """Pour chaque coin : est-il dans l'image (à plus de BORDER_MARGIN_PX du bord) ?"""
    height, width = image_shape[:2]
    m = BORDER_MARGIN_PX
    return [bool(m < x < width - m and m < y < height - m) for x, y in quad]


def homography_from_corners(quad):
    """Homographie pixel -> mm à partir des 4 coins seuls (point de départ approximatif pour l'étape 3)."""
    return cv2.getPerspectiveTransform(np.asarray(quad, np.float32), BOARD_CORNERS_MM)
