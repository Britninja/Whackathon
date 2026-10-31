"""Etape 1 (sans marqueur) : où est le panneau dans l'image ?

Idée : le panneau est percé d'une grille régulière de trous. On repère toutes les petites taches sombres, on ne garde
que celles qui forment une grille régulière (une chaise ou le plafond n'en forment pas), et la zone où cette grille est
dense = le panneau. S'il y a plusieurs panneaux, on garde le plus grand ET le plus central (celui d'en face).

Fonctions principales : find_candidate_grids(image), puis best_board_mask(image, grids, permissive).
"""
import cv2
import numpy as np

BLACKHAT_KERNEL_PX = 7      # un peu plus grand qu'un trou dans l'image
DENSITY_BLUR_PX = 25        # lissage de la carte de densité de texture (méthode de secours)
# Les réglages en pixels de ce fichier sont faits pour un trou tous les ~6 à 15 px : les plus grandes photos sont
# réduites à cette taille avant de chercher le panneau (voir perception_rgb.analyze).
MAX_IMAGE_SIDE_PX = 1920


# ---------------------------------------------------------------------- les trous et leur écart

def find_dark_spots(image):
    """Petites taches sombres (trous possibles). Retourne leurs centres (N, 2) en pixels."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_KERNEL_PX, BLACKHAT_KERNEL_PX))
    darkness = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)      # fort là où c'est plus sombre que les alentours
    spots = ((darkness > max(6, float(np.percentile(darkness, 92)))) * 255).astype(np.uint8)
    _, _, stats, centers = cv2.connectedComponentsWithStats(spots)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return centers[1:][(areas >= 2) & (areas <= 40)].astype(np.float32)


def subsample(points, max_count=20000):
    """Au-delà de max_count points, on en tire un échantillon (toujours le même) pour limiter le temps de calcul."""
    if len(points) > max_count:
        return points[np.random.default_rng(0).choice(len(points), max_count, replace=False)]
    return points


def nearest_neighbor_distances(points, chunk=800):
    """Pour chaque point, distance à son plus proche voisin (calculé par paquets pour limiter la mémoire)."""
    result = np.empty(len(points), np.float32)
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        dist = np.hypot(block[:, None, 0] - points[None, :, 0], block[:, None, 1] - points[None, :, 1])
        dist[np.arange(len(block)), np.arange(start, start + len(block))] = 1e9     # ignorer la distance à soi-même
        result[start:start + chunk] = dist.min(axis=1)
    return result


def guess_hole_spacings(spots, max_count=5):
    """Ecarts entre trous probables (en pixels) : pics de l'histogramme des distances au plus proche voisin.
    Des panneaux vus de biais ou à des distances différentes n'ont pas le même écart : on renvoie plusieurs candidats."""
    if len(spots) < 200:
        return []
    spots = subsample(spots)
    hist, edges = np.histogram(nearest_neighbor_distances(spots), bins=np.arange(3, 60, 0.5))
    smooth = np.convolve(hist, np.ones(3) / 3, mode="same")
    peaks = [k for k in range(1, len(smooth) - 1)
             if smooth[k] >= smooth[k - 1] and smooth[k] >= smooth[k + 1] and smooth[k] >= 0.08 * smooth.max()]
    peaks.sort(key=lambda k: -smooth[k])
    spacings = []
    for k in peaks:
        spacing = float(edges[k] + 0.25)
        if all(abs(spacing - other) > 0.2 * other for other in spacings):     # pas deux candidats presque égaux
            spacings.append(spacing)
        if len(spacings) >= max_count:
            break
    return spacings


def keep_grid_holes(spots, spacing, chunk=800, max_ratio=1.5, min_ratio=0.6):
    """Garde seulement les taches qui appartiennent à une grille régulière (le panneau) et rejette le reste.

    Un trou de la grille a au moins 3 voisins à environ `spacing` pixels, dont deux opposés (180°) et deux à angle
    droit (90°). max_ratio > 1.3 car vue de biais la grille n'a pas le même écart en x et en y.
    Retourne (trous retenus, spacing)."""
    if len(spots) < 200:
        return spots[:0], None
    spots = subsample(spots)
    keep = np.zeros(len(spots), bool)
    for start in range(0, len(spots), chunk):
        dx = spots[None, :, 0] - spots[start:start + chunk, None, 0]
        dy = spots[None, :, 1] - spots[start:start + chunk, None, 1]
        dist = np.hypot(dx, dy)
        is_neighbor = (dist >= min_ratio * spacing) & (dist <= max_ratio * spacing)
        for row in np.nonzero(is_neighbor.sum(axis=1) >= 3)[0]:
            nb = np.nonzero(is_neighbor[row])[0]
            angles = np.arctan2(dy[row, nb], dx[row, nb])
            angle_diff = np.abs(np.angle(np.exp(1j * (angles[:, None] - angles[None, :]))))
            has_opposite = (np.abs(angle_diff - np.pi) < 0.45).any()
            has_perpendicular = (np.abs(angle_diff - np.pi / 2) < 0.5).any()
            if has_opposite and has_perpendicular:
                keep[start + row] = True
    return spots[keep], spacing


def find_candidate_grids(image):
    """Les grilles possibles dans l'image : {écart en px: (trous de cette grille, écart)}.
    Un écart candidat = une grille possible. Sans aucun candidat : {None: (aucun trou, None)}, et best_board_mask
    passera alors à la méthode de secours (densité de texture)."""
    spots = find_dark_spots(image)
    spacings = guess_hole_spacings(spots)
    spacings = spacings + [1.2 * s for s in spacings[:2]]         # vue de biais : l'écart vertical est plus grand
    if not spacings:
        return {None: (spots[:0], None)}
    return {s: keep_grid_holes(spots, s) for s in spacings}


# ---------------------------------------------------------------------- du nuage de trous au masque du panneau

def grid_density_map(image_shape, grid):
    """Carte lissée de densité des trous de la grille (donc seulement là où est le panneau). None si pas de grille."""
    holes, spacing = grid
    if spacing is None or len(holes) < 200:
        return None
    density = np.zeros(image_shape[:2], np.float32)
    x = np.clip(holes[:, 0].astype(int), 0, density.shape[1] - 1)
    y = np.clip(holes[:, 1].astype(int), 0, density.shape[0] - 1)
    density[y, x] = 255.0
    window = max(9, int(round(3.5 * spacing)))           # fenêtre de lissage ~ quelques trous
    return cv2.GaussianBlur(cv2.blur(density, (window, window)), (0, 0), 3)


def texture_density_map(image):
    """Méthode de secours (pas de grille trouvée) : densité de petites taches sombres, sans vérifier qu'elles forment
    une grille."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BLACKHAT_KERNEL_PX, BLACKHAT_KERNEL_PX))
    darkness = np.clip(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32) * 4, 0, 255)
    return cv2.GaussianBlur(cv2.blur(darkness, (DENSITY_BLUR_PX, DENSITY_BLUR_PX)), (0, 0), 3)


def centrality_weight(contour, image_shape):
    """1 si le contour est au centre de l'image, tend vers 0 quand il s'en éloigne."""
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return 0.0
    height, width = image_shape[:2]
    dist = np.hypot(moments["m10"] / moments["m00"] - width / 2, moments["m01"] / moments["m00"] - height / 2)
    return float(np.exp(-(dist / (0.45 * width)) ** 2))


def main_region(binary):
    """Nettoie un masque binaire et garde la région la plus grande ET la plus centrale (le panneau d'en face, pas un
    voisin sur le côté), bouchée par son enveloppe convexe. Retourne (masque rempli, contour) ou (None, None)."""
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    best = max(contours, key=lambda c: cv2.contourArea(c) * centrality_weight(c, binary.shape))
    if cv2.contourArea(best) < 0.03 * binary.shape[0] * binary.shape[1]:
        return None, None
    best = cv2.convexHull(best)          # un panneau est convexe : comble les creux (prises, robot, ombres)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, [best], -1, 255, -1)
    return filled, best


def board_mask(image, grid, permissive=False):
    """Masque du panneau pour UNE grille candidate. Retourne (masque, contour, carte de densité, seuil utilisé).

    Utilise la densité des trous de la grille ; si aucune grille, la densité de texture (secours).
    permissive=True : seuils plus bas, pour un panneau vu très en biais (densité très inégale)."""
    density = grid_density_map(image.shape, grid)
    from_grid = density is not None
    if not from_grid:
        density = texture_density_map(image)
    min_threshold_1, min_threshold_2 = (0.3, 0.2) if from_grid else (6.0, 4.0)
    percentile = 98 if from_grid else 90
    factor_1 = 0.15 if permissive else 0.5
    # 1re passe : seuil relatif à une valeur haute de la densité (dans le "plateau" du panneau)
    threshold = max(min_threshold_1 * (0.4 if permissive else 1), factor_1 * float(np.percentile(density, percentile)))
    mask, contour = main_region(((density > threshold) * 255).astype(np.uint8))
    if mask is None:
        return None, None, density, threshold
    # 2e passe : seuil relatif à la densité médiane À L'INTÉRIEUR du panneau (bord plus net)
    interior = cv2.erode(mask, np.ones((31, 31), np.uint8)) > 0
    if interior.sum() > 500:
        threshold = max(min_threshold_2 * (0.4 if permissive else 1),
                        (0.12 if permissive else 0.35) * float(np.median(density[interior])))
        mask_2, contour_2 = main_region(((density > threshold) * 255).astype(np.uint8))
        if mask_2 is not None:
            mask, contour = mask_2, contour_2
    return mask, contour, density, threshold


def best_board_mask(image, grids, permissive=False):
    """Essaie chaque grille candidate et garde celle qui donne le panneau le plus grand ET le plus central.
    Retourne (grille retenue, masque, contour, carte de densité, seuil) ; masque = None si aucun panneau."""
    best = None
    for grid in grids.values():
        candidate = board_mask(image, grid, permissive)
        score = 0 if candidate[0] is None else int(cv2.contourArea(candidate[1]) * centrality_weight(candidate[1], image.shape))
        if best is None or score > best[0]:
            best = (score, grid, candidate)
    _, grid, (mask, contour, density, threshold) = best
    return grid, mask, contour, density, threshold
