"""Détection des prises d'escalade en RGB seul (aucun ArUco, aucune profondeur).

Principe
  1. Contour du panneau : la grille de trous (52 x 93, pas 15 mm) donne une texture que rien d'autre
     n'a aussi dense. On la repère (black-hat), on garde la plus grande zone = le panneau en face.
  2. Quadrilatère du panneau -> homographie -> vue de face du panneau en mm.
  3. Couleurs (HSV) cherchées UNIQUEMENT dans cette vue de face : on ne voit ni le mur d'escalade du fond,
     ni les panneaux voisins, et la taille d'une prise est la même partout (en mm).

Usage
  python perception_rgb.py              # caméra RealSense (couleur seule)
  python perception_rgb.py photo.png     # test hors-ligne sur une image
Touches : q quitter, s afficher le tableau des prises, clic gauche sur "Camera" : HSV du pixel.
"""
import sys
import cv2
import numpy as np

# ------------------------------------------------------------------ géométrie (mm)
HOLE_PITCH = 15.0
N_COLS, N_ROWS = 52, 93
# Zone texturée = une cellule de 15 mm par trou. Repère mur : origine en bas à gauche, x droite, y haut.
TEX_W, TEX_H = N_COLS * HOLE_PITCH, N_ROWS * HOLE_PITCH      # 780 x 1395 mm

# Décalage (mm) entre le centre de la tache de couleur d'une prise et le milieu de ses deux boulons (col + 1.5, row + 0.5).
# Ces valeurs-ci servent au SIMULATEUR (calibrées dessus). Sur le vrai mur, chaque couleur a la sienne : voir "dxdy" dans COLORS.
HOLD_DX, HOLD_DY = -0.3, -13.0
# Parallaxe (essai, désactivée : sur la vraie photo elle n'a pas amélioré le résultat) : hauteur du centre visible d'une prise
# au-dessus du mur, en vertical (HOLD_Z) et en horizontal (HOLD_ZX). 0 = pas de correction.
HOLD_Z = 0.0
HOLD_ZX = 0.0

# Coin bas-gauche de la zone texturée dans le repère mur (x droite, y haut, z hors du mur), en mm :
# le simulateur place le trou (0,0) en (17.5, 10) ; la cellule de 15 mm de ce trou commence donc en (10, 2.5).
TEX_X0, TEX_Y0 = 10.0, 2.5

# Vue de face : pixels par mm
S = 0.8
RECT_W, RECT_H = int(round(TEX_W * S)), int(round(TEX_H * S))

# ------------------------------------------------------------------ couleurs des prises (HSV OpenCV)
# Vrai mur, mesuré sur une photo RealSense D435i en salle très lumineuse (HSV OpenCV, H 0-179) :
#   orange : H 6-9,  S 180-200, V 110-120   |   bleu : H 97-99, S 190-220, V 80-95
#   vert (pâle, presque translucide) : H 38-54, S 40-80, V 80-130   |   bois du panneau : H 21-31, S 40-57
# Les plages sont larges autour de ces mesures ; si l'éclairage change, clique sur des prises dans live_cameras.py
# pour lire leur HSV et ajuste ici.
COLORS = {
    # "dxdy" : position du centroïde de la tache par rapport au milieu des deux boulons (mm), mesuré sur la vraie photo
    # (prises isolées, N2 : orange -3.1 / -14.6, bleu -3.0 / -14.3, écart-type ~2 mm)
    "Orange": {"lower": np.array([0, 120, 50]),   "upper": np.array([14, 255, 255]),  "bgr": (0, 140, 255), "dxdy": (-3.1, -14.6)},
    "Vert":   {"lower": np.array([36, 30, 60]),   "upper": np.array([75, 200, 255]),  "bgr": (0, 255, 0),   "dxdy": (-3.0, -14.5),
              "amax": -5},     # vert pâle : le bois est presque de la même teinte ; le vert a a* (Lab) <= -6, le bois >= -3
    "Bleu":   {"lower": np.array([90, 110, 40]),  "upper": np.array([112, 255, 255]), "bgr": (255, 150, 0), "dxdy": (-3.0, -14.3)},
}
# Simulateur MuJoCo : toutes les prises ont la même couleur (rouge-brun sombre, H~9, S~210, V 40-105),
# le bois est à H~19, S~90 : on les sépare par la teinte et la saturation.
SIM_COLORS = {
    "Prise": {"lower": np.array([3, 130, 25]), "upper": np.array([15, 255, 255]), "bgr": (0, 140, 255)},
}
HOLD_MIN_MM2, HOLD_MAX_MM2 = 300.0, 4000.0     # surface d'une prise vue de face

# ------------------------------------------------------------------ réglages du masque panneau
BH_KERNEL = 7          # black-hat : un peu plus grand qu'un trou vu dans l'image (px)
DENS_BLUR = 25         # lissage de la densité de trous (px)
BORDER_MARGIN = 8      # un coin à moins de N px du bord de l'image n'est pas un vrai coin


def _hole_candidates(frame):
    """Petites taches sombres (trous possibles) : centres (N, 2) en pixels."""
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BH_KERNEL, BH_KERNEL)))
    m = ((bh > max(6, float(np.percentile(bh, 92)))) * 255).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    ok = (st[1:, cv2.CC_STAT_AREA] >= 2) & (st[1:, cv2.CC_STAT_AREA] <= 40)
    return cen[1:][ok].astype(np.float32)


def _nn_distances(pts, chunk=800):
    nn = np.empty(len(pts), np.float32)
    for a in range(0, len(pts), chunk):
        d = np.hypot(pts[a:a + chunk, None, 0] - pts[None, :, 0], pts[a:a + chunk, None, 1] - pts[None, :, 1])
        d[np.arange(d.shape[0]), np.arange(a, a + d.shape[0])] = 1e9
        nn[a:a + chunk] = d.min(axis=1)
    return nn


def pitch_candidates(pts, max_n=5):
    """Pas de grille probables (px) : pics de l'histogramme des distances au plus proche voisin. Plusieurs panneaux vus de
    biais, ou à des distances différentes, ont des pas différents : on les essaie tous."""
    if len(pts) < 200:
        return []
    if len(pts) > 20000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
    hist, edges = np.histogram(_nn_distances(pts), bins=np.arange(3, 60, 0.5))
    h = np.convolve(hist, np.ones(3) / 3, mode="same")
    peaks = [k for k in range(1, len(h) - 1) if h[k] >= h[k - 1] and h[k] >= h[k + 1] and h[k] >= 0.08 * h.max()]
    peaks.sort(key=lambda k: -h[k])
    out = []
    for k in peaks:
        p = float(edges[k] + 0.25)
        if all(abs(p - q) > 0.2 * q for q in out):
            out.append(p)
        if len(out) >= max_n:
            break
    return out


def lattice_holes(pts, chunk=800, p=None, hi=1.5, lo=0.6):
    """Garde les taches qui font partie d'une grille régulière (le panneau) et rejette le reste (fouillis de la salle).

    Un trou de la grille a >= 3 voisins au pas p, dont deux opposés (180 deg) et deux perpendiculaires (90 deg). Sans p
    donné, on prend le pic principal de l'histogramme des distances. Retourne (points retenus, pas en pixels), ou
    (vide, None)."""
    if len(pts) < 200:
        return pts[:0], None
    if len(pts) > 20000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
    if p is None:
        cands = pitch_candidates(pts, 1)
        if not cands:
            return pts[:0], None
        p = cands[0]
    keep = np.zeros(len(pts), bool)
    for a in range(0, len(pts), chunk):
        dx, dy = pts[None, :, 0] - pts[a:a + chunk, None, 0], pts[None, :, 1] - pts[a:a + chunk, None, 1]
        d = np.hypot(dx, dy)
        near = (d >= lo * p) & (d <= hi * p)      # hi > 1.3 : la grille vue de biais n'a pas le même pas en x et en y
        for r in np.nonzero(near.sum(axis=1) >= 3)[0]:
            j = np.nonzero(near[r])[0]
            ang = np.arctan2(dy[r, j], dx[r, j])
            diff = np.abs(np.angle(np.exp(1j * (ang[:, None] - ang[None, :]))))
            if (np.abs(diff - np.pi) < 0.45).any() and (np.abs(diff - np.pi / 2) < 0.5).any():
                keep[a + r] = True
    return pts[keep], p


def lattice_density(frame, lat=None):
    """Densité locale de trous qui font partie d'une grille régulière (donc seulement le panneau), lissée.
    Retourne (carte, nombre de trous retenus) ; None si aucune grille n'est trouvée. lat = (points, pas) déjà calculés."""
    pts, p = lat if lat is not None else lattice_holes(_hole_candidates(frame))
    if p is None or len(pts) < 200:
        return None, 0
    m = np.zeros(frame.shape[:2], np.float32)
    xi, yi = np.clip(pts[:, 0].astype(int), 0, m.shape[1] - 1), np.clip(pts[:, 1].astype(int), 0, m.shape[0] - 1)
    m[yi, xi] = 255.0
    k = max(9, int(round(3.5 * p)))                 # fenêtre de lissage ~ quelques trous
    return cv2.GaussianBlur(cv2.blur(m, (k, k)), (0, 0), 3), len(pts)


def texture_density(frame):
    """Densité locale de trous : black-hat (petites taches sombres) puis lissage."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BH_KERNEL, BH_KERNEL)))
    bh = np.clip(bh.astype(np.float32) * 4, 0, 255)
    return cv2.GaussianBlur(cv2.blur(bh, (DENS_BLUR, DENS_BLUR)), (0, 0), 3)


def _centrality(c, shape):
    M = cv2.moments(c)
    if M["m00"] <= 0:
        return 0.0
    d = np.hypot(M["m10"] / M["m00"] - shape[1] / 2, M["m01"] / M["m00"] - shape[0] / 2)
    return float(np.exp(-(d / (0.45 * shape[1])) ** 2))


def _largest_filled(m):
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    h_, w_ = m.shape[:2]

    def score(c):       # grande ET centrale : le mur d'en face, pas le panneau voisin sur le côté
        M = cv2.moments(c)
        if M["m00"] <= 0:
            return 0.0
        d = np.hypot(M["m10"] / M["m00"] - w_ / 2, M["m01"] / M["m00"] - h_ / 2)
        return cv2.contourArea(c) * float(np.exp(-(d / (0.45 * w_)) ** 2))
    c = max(cnts, key=score)
    if cv2.contourArea(c) < 0.03 * m.shape[0] * m.shape[1]:
        return None, None
    c = cv2.convexHull(c)            # un panneau est convexe : comble les creux (prises, robot, ombres)
    out = np.zeros_like(m)
    cv2.drawContours(out, [c], -1, 255, -1)
    return out, c


def board_mask(frame, lat=None, soft=False):
    """Masque du panneau, carte de densité, seuil à mi-plateau.

    D'abord par la grille régulière de trous (rejette tout ce qui n'est pas la grille : plafond, chaises, panneaux voisins
    vus de biais, personnes) ; si aucune grille n'est trouvée, par la simple densité de texture (ancienne méthode)."""
    dens, n_lat = lattice_density(frame, lat)
    lattice = dens is not None
    if not lattice:
        dens = texture_density(frame)
    floor1, floor2 = (0.3, 0.2) if lattice else (6.0, 4.0)
    # statistique robuste : le panneau est une zone dense ; le centile est dans son plateau (sans les plaques ArUco du sim)
    q = 98 if lattice else 90
    k1 = 0.15 if soft else 0.5      # soft : panneau vu de biais, très inégalement dense
    thr = max(floor1 * (0.4 if soft else 1), k1 * float(np.percentile(dens, q)))
    mask, c = _largest_filled(((dens > thr) * 255).astype(np.uint8))
    if mask is None:
        return None, None, dens, thr
    # 2e passe : le seuil exact est la moitié du plateau de densité DANS le panneau (bord net, sans biais)
    inner = cv2.erode(mask, np.ones((31, 31), np.uint8)) > 0
    if inner.sum() > 500:
        thr = max(floor2 * (0.4 if soft else 1), (0.12 if soft else 0.35) * float(np.median(dens[inner])))
        m2, c2 = _largest_filled(((dens > thr) * 255).astype(np.uint8))
        if m2 is not None:
            mask, c = m2, c2
    return mask, c, dens, thr


def order_corners(pts):
    """TL, TR, BR, BL : deux points les plus hauts / les deux plus bas, puis tri en x."""
    pts = np.asarray(pts, np.float32)
    by_y = pts[np.argsort(pts[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bot = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.array([top[0], top[1], bot[1], bot[0]], np.float32)


def board_quad(contour):
    """Quadrilatère grossier du panneau : rectangle d'aire minimale de l'enveloppe convexe du contour. Robuste aux
    coins grignotés (plaques ArUco du simulateur) ; la perspective le laisse de travers de quelques pixels, ce que
    quad_from_contour puis refine_quad corrigent."""
    hull = cv2.convexHull(contour)
    if len(hull) < 4:
        return None
    return order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))


def quad_from_contour(contour, quad, band=90.0):
    """Reprend chaque côté du quadrilatère grossier sur le contour réel du masque : les points du contour proches du
    côté (milieu du côté seulement, loin des coins que les plaques ArUco du simulateur grignotent) donnent une
    droite robuste (Huber) ; les coins sont les intersections. Corrige un quadrilatère d'enveloppe convexe de travers."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    lines = []
    for i in range(4):
        p0, p1 = quad[i].astype(np.float64), quad[(i + 1) % 4].astype(np.float64)
        d = p1 - p0
        L = np.linalg.norm(d)
        if L < 20:
            return quad
        d /= L
        n = np.array([d[1], -d[0]])
        t = (pts - p0) @ d / L
        dist = np.abs((pts - p0) @ n)
        sel = pts[(t > 0.2) & (t < 0.8) & (dist < band)]
        if len(sel) < 20:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    out = []
    for i in range(4):
        (pa, da), (pb, db) = lines[(i - 1) % 4], lines[i]
        A = np.array([da, -db]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return quad
        tt = np.linalg.solve(A, pb - pa)
        out.append(pa + tt[0] * da)
    return np.array(out, np.float32)


def refine_quad(dens, thr, quad, half_window=35, n_samples=40):
    """Chaque côté du quadrilatère grossier est recalé : on cherche, le long de sa normale, le point où la
    densité de trous passe sous le seuil (précision sous le pixel), puis on ajuste une droite et on
    intersecte les droites voisines. Corrige l'arrondi des coins du masque."""
    centre = quad.mean(axis=0)
    lines = []
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 20:
            return quad
        d /= L
        n = np.array([d[1], -d[0]], np.float32)
        if np.dot(n, (p0 + p1) / 2 - centre) < 0:
            n = -n                                              # normale vers l'extérieur
        ts = np.linspace(0.2, 0.8, n_samples, dtype=np.float32)      # on évite les coins (plaques ArUco blanches sur le vrai mur)
        offs = np.arange(-half_window, half_window + 1, dtype=np.float32)
        base = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        pts = base[:, None, :] + offs[None, :, None] * n[None, None, :]
        prof = cv2.remap(dens, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        found = []
        for k in range(n_samples):
            row = prof[k]
            inside = np.where(row[: half_window + 8] >= thr)[0]
            if len(inside) == 0:
                continue
            # dernier point "dans le panneau" avant la sortie, puis interpolation du franchissement
            above = row >= thr
            idx = None
            for j in range(len(row) - 1, 0, -1):
                if above[j - 1] and not above[j]:
                    idx = j
                    break
            if idx is None:
                continue
            a, b = row[idx - 1], row[idx]
            frac = (a - thr) / (a - b + 1e-6)
            o = offs[idx - 1] + frac
            found.append(base[k] + o * n)
        if len(found) < 10:
            return quad
        f = np.array(found, np.float32)
        vx, vy, x0, y0 = cv2.fitLine(f, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    out = []
    for i in range(4):                                          # coin i = côté (i-1) ∩ côté i
        (pa, da), (pb, db) = lines[(i - 1) % 4], lines[i]
        A = np.array([da, -db]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return quad
        t = np.linalg.solve(A, pb - pa)
        out.append(pa + t[0] * da)
    return np.array(out, np.float32)


def quad_valid(quad):
    """Cadre plausible : convexe, angles intérieurs entre 45° et 135° (la perspective les déforme un peu)."""
    q = np.asarray(quad, np.float64)
    sign = []
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sign.append(np.sign(cross))
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        if not 45 < np.degrees(np.arccos(np.clip(cosang, -1, 1))) < 135:
            return False
    if len(set(sign)) != 1:
        return False
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return 0.33 < w / h < 0.85                 # attendu ~0.56 ; une simple bande de prises donnerait ~0.24


def border_cuts(mask):
    """Côtés de l'image que le masque du panneau touche : le panneau y est coupé (pas de vrai bord)."""
    m = BORDER_MARGIN
    bands = {"gauche": mask[:, :m], "droite": mask[:, -m:], "haut": mask[:m, :], "bas": mask[-m:, :]}
    return [n for n, b in bands.items() if np.count_nonzero(b) > 40]


def pose_from_points(obj_m, img_px, K):
    """solvePnP planaire (IPPE) puis affinage LM. obj_m : points du mur (m, repère mur, z = 0), img_px : pixels.
    Retourne (rvec, tvec, erreur de reprojection moyenne en px, position caméra dans le repère mur en m)."""
    obj = np.asarray(obj_m, np.float64).reshape(-1, 3)
    img = np.asarray(img_px, np.float64).reshape(-1, 1, 2)
    Kf = np.asarray(K, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, Kf, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    if len(obj) > 8:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, Kf, None, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, Kf, None)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1).mean())
    R, _ = cv2.Rodrigues(rvec)
    return rvec.ravel(), tvec.ravel(), err, (-R.T @ tvec).ravel()


def estimate_camera_pose(quad, K):
    """Pose à partir des 4 seuls coins du panneau : sensible au bruit (1 px de bruit sur les coins = ~3 cm et ~0,7°
    d'erreur). Préférer la pose par la grille de trous (res["pose"] quand la grille est reconnue)."""
    obj = np.array([[TEX_X0, TEX_Y0 + TEX_H, 0], [TEX_X0 + TEX_W, TEX_Y0 + TEX_H, 0],
                    [TEX_X0 + TEX_W, TEX_Y0, 0], [TEX_X0, TEX_Y0, 0]], np.float64) / 1000.0   # TL TR BR BL, en m
    return pose_from_points(obj, quad, K)


def corners_inside(quad, shape):
    h, w = shape[:2]
    m = BORDER_MARGIN
    return [bool(m < x < w - m and m < y < h - m) for x, y in quad]


def quad_homography(quad):
    """Homographie pixel -> repère de la zone texturée (mm, origine en bas à gauche, y vers le haut)."""
    dst_mm = np.array([[0, TEX_H], [TEX_W, TEX_H], [TEX_W, 0], [0, 0]], np.float32)  # TL TR BR BL
    return cv2.getPerspectiveTransform(np.asarray(quad, np.float32), dst_mm)


def rectify(frame, H_px2mm):
    """Vue de face du panneau (S pixels par mm) à partir de l'homographie pixel -> mm."""
    to_rect = np.array([[S, 0, 0], [0, -S, TEX_H * S], [0, 0, 1]], np.float64)       # mm -> pixels de la vue
    H_px2rect = to_rect @ H_px2mm
    rect = cv2.warpPerspective(frame, H_px2rect, (RECT_W, RECT_H), flags=cv2.INTER_AREA)
    return rect, H_px2mm, H_px2rect


def _circular_shift(frac):
    """Décalage moyen (dans [-0.5, 0.5)) d'une série de fractions de pas, robuste au repliement."""
    ang = 2 * np.pi * np.asarray(frac)
    return float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi))


def detect_hole_centers(frame, mask, scale, H0=None):
    """Centres des trous (petites taches sombres) dans le panneau, en pixels. scale = pixels par mm."""
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    k = int(round(8 * scale)) | 1
    k = max(k, 5)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    inner = cv2.erode(mask, np.ones((int(2 * scale) | 1,) * 2, np.uint8))
    if H0 is not None:                                    # coins (150 mm) exclus : plaques ArUco du simulateur
        Hinv = np.linalg.inv(H0)
        for cx in (0.0, TEX_W):
            for cy in (0.0, TEX_H):
                sx, sy = (1 if cx == 0 else -1), (1 if cy == 0 else -1)
                sq = np.float32([[cx, cy], [cx + sx * 150, cy], [cx + sx * 150, cy + sy * 150], [cx, cy + sy * 150]])
                px = cv2.perspectiveTransform(sq.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
                cv2.fillConvexPoly(inner, px.astype(np.int32), 0)
    inner = inner > 0
    vals = bh[inner]
    if vals.size == 0:
        return np.zeros((0, 2), np.float32)
    thr = max(6.0, 0.5 * float(np.percentile(vals, 90)))
    m = ((bh > thr) & inner).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    amin, amax = 0.5 * (1.6 * scale) ** 2 * np.pi, 4.0 * (3.2 * scale) ** 2
    keep = [i for i in range(1, n) if max(amin, 2) <= st[i, cv2.CC_STAT_AREA] <= amax]
    return cen[keep].astype(np.float32)


def _window_start(counts, size, offset=4, max_shift=3):
    """Début (en cases, entre -max_shift et +max_shift) de la fenêtre de `size` colonnes (ou lignes) qui contient le
    plus de trous, en pénalisant les colonnes peuplées juste en dehors : une fenêtre décalée d'une case perd une
    colonne peuplée (le bord de la grille) et en laisse une dehors. A égalité, on garde 0. `counts` est décalé de
    `offset` (counts[k] = nombre de trous de l'indice k - offset)."""
    best, best_score = 0, -1e18
    order = [0] + [d for n in range(1, max_shift + 1) for d in (-n, n)]
    for start in order:
        k = start + offset
        if k < 1 or k + size >= len(counts):
            continue
        score = counts[k:k + size].sum() - counts[k - 1] - counts[k + size]
        if score > best_score:
            best, best_score = start, score
    return best


def index_lattice(pts, H0, tol=0.3, max_jump=3):
    """Numérote les trous d'une grille régulière en partant du trou central et en avançant de voisin en voisin (comme on
    suit les lignes d'un quadrillage), sans exiger que la grille soit une homographie parfaite : la base locale
    (vecteur d'un trou au suivant, à droite et en haut) est suivie et mise à jour pas à pas. Un trou caché par une prise
    est franchi en sautant de 2 ou 3 pas. Retourne (indices (n,2) entiers, ou -99999 pour un trou non numéroté)."""
    n = len(pts)
    Hinv = np.linalg.inv(H0)
    c_px = cv2.perspectiveTransform(np.float32([[[TEX_W / 2, TEX_H / 2]]]), Hinv)[0, 0]
    # graine : un trou entouré de ses 4 voisins (pas un trou isolé au milieu des prises), le plus proche du centre
    pitch = float(np.mean([np.hypot(*(cv2.perspectiveTransform(np.float32([[[TEX_W / 2 + dx, TEX_H / 2 + dy]]]), Hinv)[0, 0] - c_px))
                           for dx, dy in ((HOLE_PITCH, 0), (0, HOLE_PITCH))]))
    nb = np.zeros(n, np.int32)
    for a in range(0, n, 800):
        d = np.hypot(pts[a:a + 800, None, 0] - pts[None, :, 0], pts[a:a + 800, None, 1] - pts[None, :, 1])
        nb[a:a + 800] = ((d >= 0.75 * pitch) & (d <= 1.3 * pitch)).sum(axis=1)
    cand = np.nonzero(nb >= 4)[0]
    if len(cand) == 0:
        cand = np.nonzero(nb >= 3)[0]
    if len(cand) == 0:
        return np.full((n, 2), -99999, np.int64)
    seed = int(cand[np.argmin(np.hypot(pts[cand, 0] - c_px[0], pts[cand, 1] - c_px[1]))])
    p0 = pts[seed]
    m0 = cv2.perspectiveTransform(np.float32([[p0]]), H0)[0, 0]
    a1_0 = cv2.perspectiveTransform(np.float32([[m0 + [HOLE_PITCH, 0]]]), Hinv)[0, 0] - p0
    a2_0 = cv2.perspectiveTransform(np.float32([[m0 + [0, HOLE_PITCH]]]), Hinv)[0, 0] - p0
    cell = max(4.0, float(np.hypot(*a1_0)))
    grid = {}
    for k, (x, y) in enumerate(pts):
        grid.setdefault((int(x // cell), int(y // cell)), []).append(k)

    def find(target, radius):
        cx, cy = int(target[0] // cell), int(target[1] // cell)
        best, bd = -1, radius
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in grid.get((cx + dx, cy + dy), ()):
                    d = float(np.hypot(pts[k, 0] - target[0], pts[k, 1] - target[1]))
                    if d < bd:
                        best, bd = k, d
        return best

    idx = np.full((n, 2), -99999, np.int64)
    A1, A2 = np.zeros((n, 2)), np.zeros((n, 2))
    idx[seed] = (0, 0)
    A1[seed], A2[seed] = a1_0, a2_0
    queue, taken = [seed], {(0, 0): seed}
    while queue:
        k = queue.pop()
        i, j = idx[k]
        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
            a = (A1[k] if axis == 0 else A2[k]) * sign
            for step in range(1, max_jump + 1):
                di, dj = (sign * step, 0) if axis == 0 else (0, sign * step)
                key = (int(i + di), int(j + dj))
                if key in taken:
                    break
                q = find(pts[k] + a * step, tol * float(np.hypot(*a)))
                if q < 0 or idx[q, 0] != -99999:
                    continue
                idx[q] = key
                taken[key] = q
                vec = (pts[q] - pts[k]) / (step * sign)
                A1[q] = 0.6 * A1[k] + 0.4 * vec if axis == 0 else A1[k]
                A2[q] = 0.6 * A2[k] + 0.4 * vec if axis == 1 else A2[k]
                queue.append(q)
                break
    return idx


def lattice_refine_indexed(H0, pts):
    """Homographie pixel -> mm (repère texturé) par numérotation de proche en proche des trous (voir index_lattice).
    pts : trous de la grille (pixels), déjà filtrés. Retourne comme lattice_refine, ou None."""
    if len(pts) < 200:
        return None
    idx = index_lattice(pts, H0)
    ok = idx[:, 0] != -99999
    if ok.sum() < 200:
        return None
    ii, jj = idx[ok, 0], idx[ok, 1]
    cc = np.bincount(ii - ii.min()); rr = np.bincount(jj - jj.min())

    def start(counts, size):
        best, bs = None, -1e18
        for s0 in range(0, len(counts) - size + 1) if len(counts) >= size else [0]:
            sc = counts[s0:s0 + size].sum() - (counts[s0 - 1] if s0 > 0 else 0) - (counts[s0 + size] if s0 + size < len(counts) else 0)
            if sc > bs:
                best, bs = s0, sc
        return best
    if len(cc) < N_COLS - 1 or len(rr) < N_ROWS - 1:
        return None
    i0, j0 = ii.min() + start(cc, N_COLS), jj.min() + start(rr, N_ROWS)
    sel = (ii >= i0) & (ii < i0 + N_COLS) & (jj >= j0) & (jj < j0 + N_ROWS)
    px = pts[ok][sel]
    nodes = np.stack([(ii[sel] - i0 + 0.5) * HOLE_PITCH, (jj[sel] - j0 + 0.5) * HOLE_PITCH], 1).astype(np.float32)
    if len(px) < 200:
        return None
    H, inl = cv2.findHomography(px, nodes, cv2.RANSAC, 2.5)
    if H is None:
        return None
    for tl in (2.0, 1.5):                              # on resserre : refit sur les inliers
        mm = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)
        good = np.hypot(*(mm - nodes).T) < tl
        if good.sum() < 150:
            return None
        H, _ = cv2.findHomography(px[good], nodes[good], 0)
    mm = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)
    good = np.hypot(*(mm - nodes).T) < 2.0
    return H, px[good], nodes[good], int(good.sum())


def lattice_refine(frame, mask, H0, scale):
    """Affine l'homographie pixel -> mm avec la grille de trous (52 x 93, pas 15 mm) : des milliers de points
    connus au lieu de 4 coins. Retourne (H, pts_px, nodes_mm (repère texturé), nombre d'inliers) ou None."""
    pts = detect_hole_centers(frame, mask, scale, H0)
    if len(pts) < 200:
        return None
    # les coins du panneau (150 mm) sont écartés : sur le mur du simulateur les plaques ArUco y cachent des trous et
    # leurs carrés noirs ressemblent à des trous ; sans marqueurs on perd juste ~80 trous sur 4836 par coin
    m0 = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H0).reshape(-1, 2)
    near_x = (m0[:, 0] < 150) | (m0[:, 0] > TEX_W - 150)
    near_y = (m0[:, 1] < 150) | (m0[:, 1] > TEX_H - 150)
    pts = pts[~(near_x & near_y)]
    if len(pts) < 200:
        return None
    H = H0.copy()
    tol = (6.0, 4.0, 2.5, 1.5)                    # on resserre la tolérance à mesure que l'homographie s'affine
    off = 4
    pts_in = nodes_in = None
    for it, tl in enumerate(tol):
        mm = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        u, v = mm[:, 0] / HOLE_PITCH - 0.5, mm[:, 1] / HOLE_PITCH - 0.5
        if it == 0:                                                   # recentrage de la phase (décalage global)
            dx, dy = _circular_shift(u - np.round(u)), _circular_shift(v - np.round(v))
            u, v = u - dx, v - dy
        i, j = np.round(u), np.round(v)
        res = np.hypot(u - i, v - j) * HOLE_PITCH
        ok = (res < tl) & (i >= -off) & (i < N_COLS + off) & (j >= -off) & (j < N_ROWS + off)
        if ok.sum() < 150:
            return None
        if it == 1:
            # indice absolu : la grille fait exactement 52 colonnes x 93 lignes ; on cherche la fenêtre de cette taille
            # qui contient le plus de trous (H est maintenant cohérente sur tout le panneau : les colonnes se lisent)
            cc = np.bincount((i[ok] + off).astype(int), minlength=N_COLS + 2 * off + 1)
            rr = np.bincount((j[ok] + off).astype(int), minlength=N_ROWS + 2 * off + 1)
            i0, j0 = _window_start(cc, N_COLS, off), _window_start(rr, N_ROWS, off)
            i, j = i - i0, j - j0
        if it >= 1:
            ok &= (i >= 0) & (i < N_COLS) & (j >= 0) & (j < N_ROWS)
            if ok.sum() < 150:
                return None
        nodes = np.stack([(i + 0.5) * HOLE_PITCH, (j + 0.5) * HOLE_PITCH], axis=1)[ok]
        Hn, inl = cv2.findHomography(pts[ok], nodes.astype(np.float32), cv2.RANSAC, max(1.0, tl / 3))   # seuil en mm
        if Hn is None:
            return None
        H = Hn
        keep = inl.ravel().astype(bool)
        pts_in, nodes_in = pts[ok][keep], nodes[keep]
    return H, pts_in, nodes_in, int(len(pts_in))


def to_grid(x, y, dx=None, dy=None):
    """Position en mm (repère de la zone texturée) -> (col, row) du trou gauche de la prise."""
    dx = HOLD_DX if dx is None else dx
    dy = HOLD_DY if dy is None else dy
    return int(round((x - dx) / HOLE_PITCH - 1.5)), int(round((y - dy) / HOLE_PITCH - 0.5))


def split_touching(mask):
    """Sépare les prises qui se touchent (distance transform + watershed, comme dans ton exemple de cours).
    Retourne une image d'étiquettes : 0 = fond, 1..k = une prise par étiquette."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    win = int(18 * S) | 1                                          # deux centres de prises sont à plus de 18 mm
    peaks = ((dist >= cv2.dilate(dist, np.ones((win, win), np.uint8)) - 1e-6) & (dist > 3 * S)).astype(np.uint8)
    k, seeds = cv2.connectedComponents(peaks)
    markers = seeds.astype(np.int32)
    markers[mask == 0] = k                                         # fond
    relief = cv2.cvtColor(255 - cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(relief, markers)
    markers[(markers == k) | (markers < 0)] = 0
    return markers


def _hold_template(m):
    """Forme moyenne d'une prise isolée (en pixels de la vue de face, centrée sur son centroïde), apprise sur l'image.
    Retourne (gabarit booléen, aire d'une prise, rayon) ou None s'il n'y a pas assez de prises isolées."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    areas = np.array([st[i, cv2.CC_STAT_AREA] for i in range(1, n)])
    areas = areas[areas > HOLD_MIN_MM2 * S * S]
    if len(areas) < 3:
        return None
    a1 = float(np.percentile(areas, 25))
    R = int(40 * S)
    acc, cnt = np.zeros((2 * R + 1, 2 * R + 1), np.float32), 0
    for i in range(1, n):
        if 0.8 * a1 <= st[i, cv2.CC_STAT_AREA] <= 1.25 * a1:
            cx, cy = cen[i]
            acc += cv2.warpAffine((lab == i).astype(np.float32), np.float32([[1, 0, R - cx], [0, 1, R - cy]]),
                                  (2 * R + 1, 2 * R + 1), flags=cv2.INTER_LINEAR)
            cnt += 1
    if cnt < 3:
        return None
    return (acc / cnt) > 0.5, a1, R


def _decode_blob(blob, T, R, thr=0.8, dxdy=None):
    """Plusieurs prises collées dans un même blob : on teste chaque case de la grille avec le gabarit d'une prise
    (la case est retenue si le gabarit tient dans le blob) et on garde les cases qui ne se chevauchent pas physiquement
    (deux prises ne peuvent pas être à moins de 2 colonnes et 2 lignes l'une de l'autre, hors diagonale)."""
    odx, ody = dxdy if dxdy is not None else (HOLD_DX, HOLD_DY)
    ys0, xs0 = np.nonzero(T)
    ta = float(T.sum())
    x0, y0, w, h = cv2.boundingRect(blob.astype(np.uint8))
    cands = []
    for c in range(0, N_COLS - 2):
        for r in range(0, N_ROWS):
            u = ((c + 1.5) * HOLE_PITCH + odx) * S
            v = (TEX_H - ((r + 0.5) * HOLE_PITCH + ody)) * S
            if not (x0 - R <= u <= x0 + w + R and y0 - R <= v <= y0 + h + R):
                continue
            ys, xs = ys0 - R + int(round(v)), xs0 - R + int(round(u))
            ok = (ys >= 0) & (ys < blob.shape[0]) & (xs >= 0) & (xs < blob.shape[1])
            if ok.sum() < 0.9 * ta:
                continue
            cov = blob[ys[ok], xs[ok]].sum() / ta
            if cov >= thr:
                cands.append((cov, c, r))
    cands.sort(key=lambda t: -t[0])
    kept = []
    for cov, c, r in cands:
        if all(not (abs(c - c2) <= 2 and abs(r - r2) <= 2 and not (abs(c - c2) == 2 and abs(r - r2) == 2))
               for _, c2, r2 in kept):
            kept.append((cov, c, r))
    return [((c + 1.5) * HOLE_PITCH + odx, (r + 0.5) * HOLE_PITCH + ody) for _, c, r in kept]


def detect_holds(rect, colors=None):
    """Prises dans la vue de face. Retourne [{name, bgr, x, y (mm), col, row}].
    Une prise isolée : centroïde de son blob. Plusieurs prises collées : décodage par gabarit sur la grille
    (si le gabarit n'est pas appris, on retombe sur la découpe par watershed)."""
    blur = cv2.GaussianBlur(rect, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lab_a = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.int16) - 128      # < 0 : vers le vert
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(4 * S) | 1, int(4 * S) | 1))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(5 * S) | 1, int(5 * S) | 1))
    amin, amax = HOLD_MIN_MM2 * S * S, HOLD_MAX_MM2 * S * S
    out = []

    def add(name, bgr, x, y, u=None, v=None, off=None):
        col, row = to_grid(x, y, *(off or (None, None)))
        out.append({"name": name, "bgr": bgr, "u": x * S if u is None else u, "v": (TEX_H - y) * S if v is None else v,
                    "x": x, "y": y, "col": col, "row": row, "off": off})

    pal = COLORS if colors is None else colors
    solid = None                                   # prises franches (orange, bleu) : leur liseré clair ressemble à du vert pâle
    if any("amax" in p for p in pal.values()):
        solid = np.zeros(rect.shape[:2], np.uint8)
        for q in pal.values():
            if "amax" not in q:
                solid |= cv2.inRange(hsv, q["lower"], q["upper"])
        solid = cv2.dilate(cv2.morphologyEx(solid, cv2.MORPH_OPEN, open_k), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(10 * S) | 1,) * 2))
    for name, p in pal.items():
        m = cv2.inRange(hsv, p["lower"], p["upper"])
        if "amax" in p:
            m = m & ((lab_a <= p["amax"]).astype(np.uint8) * 255)
            if solid is not None:
                m[solid > 0] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, open_k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
        tmpl = _hold_template(m)
        if tmpl is None:                                            # pas assez de prises isolées : watershed
            labels = split_touching(m)
            for i in range(1, labels.max() + 1):
                reg = labels == i
                if not (amin < int(reg.sum()) < amax):
                    continue
                ys, xs = np.nonzero(reg)
                u, v = float(xs.mean()), float(ys.mean())
                add(name, p["bgr"], u / S, TEX_H - v / S, u, v, p.get("dxdy"))
            continue
        T, a1, R = tmpl
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if a < 0.6 * a1:
                continue
            if a <= 1.35 * a1:                                      # une prise seule
                u, v = float(cen[i][0]), float(cen[i][1])
                add(name, p["bgr"], u / S, TEX_H - v / S, u, v, p.get("dxdy"))
            else:                                                   # prises collées
                for x, y in _decode_blob(lab == i, T, R, dxdy=p.get("dxdy")):
                    add(name, p["bgr"], x, y, off=p.get("dxdy"))
    return out


def nominal_K(shape):
    """Paramètres internes approximatifs d'une RealSense D435i (champ horizontal ~69,4 deg) si on n'a pas ceux de la caméra."""
    h, w = shape[:2]
    f = 0.7226 * w
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def correct_parallax(holds, pos_m, z_mm=None, zx_mm=None):
    """Une prise dépasse du mur : vue de biais, sa position sur le plan du mur est décalée. Le centre réel est sur le
    rayon caméra -> prise, à la hauteur z du mur : on le ramène vers le pied de la caméra de (z / distance caméra-mur) de
    l'écart. pos_m : position de la caméra dans le repère mur (m)."""
    z = HOLD_Z if z_mm is None else z_mm
    cx, cy, cz = pos_m[0] * 1000.0 - TEX_X0, pos_m[1] * 1000.0 - TEX_Y0, pos_m[2] * 1000.0
    if cz <= 100:
        return holds
    k = z / cz
    kx = (HOLD_ZX if zx_mm is None else zx_mm) / cz
    for h in holds:
        h["x"] += kx * (cx - h["x"])
        h["y"] += k * (cy - h["y"])
        h["col"], h["row"] = to_grid(h["x"], h["y"], *(h.get("off") or (None, None)))
    return holds


MAX_SIDE = 1920        # les réglages en pixels (noyaux, lissages) sont faits pour un pas de grille de ~6 à 15 px : on réduit les grandes photos


def analyze(frame, colors=None, K=None):
    """Tout le pipeline sur une image BGR (photo de téléphone comprise : une image de plus de 1920 px est réduite pour la
    détection du panneau et de la grille, puis le redressement et les prises sont refaits sur l'image pleine résolution).
    Ne lève jamais : renvoie un dict de résultats + statut."""
    big = max(frame.shape[:2])
    if big <= 1.15 * MAX_SIDE:
        return _analyze_core(frame, colors, K)
    sc = MAX_SIDE / big
    small = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    Ks = None
    if K is not None:
        Ks = np.asarray(K, np.float64).copy()
        Ks[:2] *= sc
    res = _analyze_core(small, colors, Ks)
    D = np.diag([sc, sc, 1.0])
    if res["quad"] is not None:
        res["quad"] = res["quad"] / sc
    if res["mask"] is not None:
        res["mask"] = cv2.resize(res["mask"], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    if res["H_px2mm"] is not None:
        Hf = res["H_px2mm"] @ D                                      # pixel plein format -> mm
        rect, _, H_rect = rectify(frame, Hf)
        res["H_px2mm"], res["rect"], res["H_px2rect"] = Hf, rect, H_rect
        if res["holds"] is not None and res["status"].startswith("OK"):
            pose = res["pose"]
            res["holds"] = detect_holds(rect, colors)
            if pose is not None and res["holds"]:
                correct_parallax(res["holds"], pose[3])
        res["status"] = res["status"].replace("OK :", "OK (photo réduite pour la grille) :") if False else res["status"]
    return res


def _analyze_core(frame, colors=None, K=None):
    """Le pipeline sur une image de taille raisonnable."""
    res = {"status": "", "mask": None, "quad": None, "inside": None, "rect": None,
           "holds": [], "H_px2rect": None, "H_px2mm": None, "pose": None, "lattice": None}
    cand_pts = _hole_candidates(frame)
    pc = pitch_candidates(cand_pts)
    pc = pc + [1.2 * p for p in pc[:2]]                      # vue de biais : le pas en y est plus grand qu'en x
    lats = {p: lattice_holes(cand_pts, p=p) for p in (pc or [None])}
    lat = None
    for soft in (False, True):          # 2e essai si la grille n'est pas reconnue : masque plus permissif (vue très oblique)
        best = None
        for p, lat_p in lats.items():    # plusieurs pas possibles : on garde la grille la plus grande ET la plus centrale
            r = board_mask(frame, lat_p, soft)
            area = 0 if r[0] is None else int(cv2.contourArea(r[1]) * _centrality(r[1], frame.shape))
            if best is None or area > best[0]:
                best = (area, lat_p, r)
        mask, contour, dens, thr = best[2]
        if mask is None:
            res["status"] = "PANNEAU NON TROUVE"
            return res
        res["mask"] = mask
        quad = board_quad(contour)
        if quad is None:
            res["status"] = "QUADRILATERE NON TROUVE"
            return res
        quad = quad_from_contour(contour, quad)   # côtés repris sur le contour réel du masque (2 passes)
        quad = quad_from_contour(contour, quad, band=40.0)
        quad = refine_quad(dens, thr, quad)       # puis recalés au sous-pixel : coins = intersections des 4 côtés
        res["quad"] = quad
        inside = corners_inside(quad, frame.shape)
        res["inside"] = inside
        cuts = border_cuts(mask)
        if cuts or not all(inside):
            what = ("bords coupés : " + ", ".join(cuts)) if cuts else "coins hors image"
            res["status"] = f"PANNEAU INCOMPLET ({what}) - recule la camera"
            return res
        if not quad_valid(quad):
            res["status"] = "CADRE INVALIDE (angles ou proportions incoherents : image trop petite ? passe en 1920x1080)"
            if soft:
                return res
            continue
        H = quad_homography(quad)
        scale = float(np.sqrt(abs(cv2.contourArea(quad.astype(np.float32))) / (TEX_W * TEX_H)))     # pixels par mm
        lat_pts = best[1][0]      # tous les trous de la grille (le quadrilatère peut être faux d'un bord : on ne s'en sert que pour démarrer)
        lat = lattice_refine_indexed(H, lat_pts)          # numérotation de proche en proche (robuste, sans bonne homographie initiale)
        if lat is None:
            lat = lattice_refine(frame, mask, H, scale)   # ancienne méthode, en secours
        if lat is not None:
            break
    res["lattice"] = None
    if lat is not None:
        H, pts_px, nodes_mm, n_in = lat
        res["lattice"] = n_in
        corners_mm = np.float32([[0, TEX_H], [TEX_W, TEX_H], [TEX_W, 0], [0, 0]]).reshape(-1, 1, 2)
        res["quad"] = cv2.perspectiveTransform(corners_mm, np.linalg.inv(H)).reshape(-1, 2)   # coins exacts de la grille
    rect, H_mm, H_rect = rectify(frame, H)
    res["rect"], res["H_px2rect"], res["H_px2mm"] = rect, H_rect, H
    res["holds"] = detect_holds(rect, colors)
    if K is None:
        K = nominal_K(frame.shape)
    if K is not None:
        if lat is not None:
            obj = np.column_stack([nodes_mm[:, 0] + TEX_X0, nodes_mm[:, 1] + TEX_Y0, np.zeros(len(nodes_mm))]) / 1000.0
            res["pose"] = pose_from_points(obj, pts_px, K)
        if res["pose"] is not None and res["holds"]:
            correct_parallax(res["holds"], res["pose"][3])
        # sans la grille de trous, la pose par les 4 seuls coins est trop bruitée (~30 cm) : on ne la donne pas
    if res["lattice"]:
        res["status"] = f"OK : {len(res['holds'])} prises (grille : {res['lattice']} trous)"
    else:
        res["status"] = f"GRILLE NON RECONNUE : {len(res['holds'])} prises mais positions non fiables, pas de pose"
    return res


def draw(frame, res):
    vis = frame.copy()
    if res["mask"] is not None:
        cnts, _ = cv2.findContours(res["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 255, 0), 1)
    if res["quad"] is not None:
        cv2.polylines(vis, [res["quad"].astype(int)], True, (0, 255, 255), 3)
        for (x, y), ok, nm in zip(res["quad"], res["inside"] or [True] * 4, ["HG", "HD", "BD", "BG"]):
            cv2.circle(vis, (int(x), int(y)), 9, (0, 255, 0) if ok else (0, 0, 255), -1)
            cv2.putText(vis, nm, (int(x) + 12, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    rect_vis = None
    if res["rect"] is not None:
        rect_vis = res["rect"].copy()
        Hinv = np.linalg.inv(res["H_px2rect"])
        for h in res["holds"]:
            cv2.circle(rect_vis, (int(h["u"]), int(h["v"])), 7, h["bgr"], 2)
            cv2.putText(rect_vis, f'({h["col"]},{h["row"]})', (int(h["u"]) + 9, int(h["v"]) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            p = cv2.perspectiveTransform(np.float32([[[h["u"], h["v"]]]]), Hinv)[0, 0]
            cv2.circle(vis, (int(p[0]), int(p[1])), 8, h["bgr"], 2)
    ok = res["status"].startswith("OK")
    cv2.putText(vis, res["status"], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if ok else (0, 0, 255), 2)
    return vis, rect_vis


def print_table(holds):
    print(f"--- {len(holds)} prises ---")
    for h in sorted(holds, key=lambda h: (h["row"], h["col"])):
        print(f'{h["name"]:7s} col={h["col"]:3d} row={h["row"]:3d}  x={h["x"]:6.0f} mm  y={h["y"]:6.0f} mm')


def main():
    if len(sys.argv) > 1:                                    # test hors-ligne sur une image
        frame = cv2.imread(sys.argv[1])
        res = analyze(frame)
        vis, rect_vis = draw(frame, res)
        print(res["status"])
        print_table(res["holds"])
        cv2.imshow("Camera", vis)
        if rect_vis is not None:
            cv2.imshow("Mur de face", rect_vis)
        cv2.waitKey(0)
        return

    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(cfg)
    state = {"hsv": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["hsv"] is not None:
            print("HSV:", state["hsv"][y, x])
    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", on_mouse)
    last = []
    try:
        while True:
            cf = pipeline.wait_for_frames().get_color_frame()
            if not cf:
                continue
            frame = np.asanyarray(cf.get_data()).copy()
            state["hsv"] = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            res = analyze(frame)
            if res["holds"]:
                last = res["holds"]
            vis, rect_vis = draw(frame, res)
            cv2.imshow("Camera", vis)
            if rect_vis is not None:
                cv2.imshow("Mur de face", rect_vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                print_table(last)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()