"""Repere metrique du panneau Whackathon : RealSense + 4 marqueurs ArUco.

Origine: coin inferieur gauche du panneau (X vers la droite, Y vers le haut,
Z sort du panneau). Toutes les distances sont en millimetres.

Les 4 feuilles sont supposees coupees sur les pointilles et placees avec
leur coin fleche exactement sur le coin correspondant du panneau.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# A adapter si les dimensions physiques du panneau sont differentes.
LARGEUR_MUR_MM = 800.0
HAUTEUR_MUR_MM = 1400.0
TAILLE_CODE_MM = 100.0       # carre noir ArUco
BORDURE_MM = 18.0          # bordure blanche sur les feuilles

LARGEUR_IMAGE = 640
HAUTEUR_IMAGE = 480
FPS = 30

FICHIER_CALIBRATION = Path(__file__).resolve().parent / "wall_calibration.json"
FENETRE = "RealSense - Repere metrique du mur"

clic = None


def carte_rectification_inverse_brown(intr):
    """Map (image redressee -> image brute) pour RealSense inverse Brown.

    Les coefficients RealSense decrivent la transformation pixel brut ->
    rayon ideal. On inverse numeriquement cette transformation pour cv2.remap.
    La carte est calculee une seule fois, jamais dans la boucle video.
    """
    hauteur, largeur = intr.height, intr.width
    v, u = np.indices((hauteur, largeur), dtype=np.float64)
    cible_x = (u - intr.ppx) / intr.fx
    cible_y = (v - intr.ppy) / intr.fy
    x, y = cible_x.copy(), cible_y.copy()
    k1, k2, p1, p2, k3 = map(float, intr.coeffs)

    for _ in range(15):
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        ux = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        uy = y * radial + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
        x += cible_x - ux
        y += cible_y - uy

    # Erreur residuelle de l'inversion (en pixels).
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    ux = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    uy = y * radial + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
    erreur = max(float(np.max(np.abs((ux-cible_x)*intr.fx))),
                 float(np.max(np.abs((uy-cible_y)*intr.fy))))
    if not np.isfinite(erreur) or erreur > 0.01:
        raise RuntimeError(f"Impossible de rectifier le modele inverse Brown: {erreur:.3f} px")

    return (x * intr.fx + intr.ppx).astype(np.float32), \
           (y * intr.fy + intr.ppy).astype(np.float32)


def souris(event, x, y, flags, userdata):
    global clic
    if event == cv2.EVENT_LBUTTONDOWN:
        clic = (x, y)


def coins_mur_marqueur(identifiant):
    """Les 4 coins du CARRE NOIR en mm (HG, HD, BD, BG)."""
    a, b = TAILLE_CODE_MM, BORDURE_MM

    if identifiant in (1, 3):
        gauche = b
    else:
        gauche = LARGEUR_MUR_MM - b - a

    if identifiant in (1, 2):
        haut = HAUTEUR_MUR_MM - b
    else:
        haut = b + a

    return np.array([
        [gauche, haut],
        [gauche + a, haut],
        [gauche + a, haut - a],
        [gauche, haut - a],
    ], dtype=np.float64)


def pixel_vers_mur(x, y, H_image_vers_mur, K, dist):
    """Convertit un pixel du flux couleur en (X, Y) du PLAN du mur (mm)."""
    point = np.array([[[float(x), float(y)]]], dtype=np.float64)
    point_corrige = cv2.undistortPoints(point, K, dist, P=K)
    resultat = cv2.perspectiveTransform(point_corrige, H_image_vers_mur)
    return tuple(float(v) for v in resultat[0, 0])


def analyse_mur(image, detecteur, K, dist):
    """Retourne H, pose, coins et erreur si les 4 ArUco sont visibles."""
    coins_detectes, ids, _ = detecteur.detectMarkers(image)
    if ids is None:
        return None, coins_detectes, []

    presents = [int(i) for i in ids.flatten()]
    if not all(i in presents for i in (1, 2, 3, 4)):
        return None, coins_detectes, presents

    img = []
    reel = []
    for identifiant, coins in zip(presents, coins_detectes):
        if identifiant not in (1, 2, 3, 4):
            continue
        # OpenCV renvoie HG, HD, BD, BG sur les marqueurs montes a l'endroit.
        img.append(coins.reshape(4, 2).astype(np.float64))
        reel.append(coins_mur_marqueur(identifiant))

    pixels = np.concatenate(img)
    mur = np.concatenate(reel)

    # Homographie calculee dans l'image corrigee de la distorsion optique.
    pixels_corriges = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), K, dist, P=K
    ).reshape(-1, 2)

    H_mur_vers_image, inliers = cv2.findHomography(
        mur, pixels_corriges, cv2.RANSAC, 3.0
    )
    if H_mur_vers_image is None or inliers is None or inliers.sum() < 12:
        return None, coins_detectes, presents

    H_image_vers_mur = np.linalg.inv(H_mur_vers_image)

    # Pose 3D du plan du mur par rapport a la camera couleur.
    points_3d = np.column_stack([mur, np.zeros(len(mur))])
    ok, rvec, tvec = cv2.solvePnP(
        points_3d, pixels, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, coins_detectes, presents

    reproj = cv2.perspectiveTransform(
        mur.reshape(-1, 1, 2), H_mur_vers_image
    ).reshape(-1, 2)
    erreur = float(np.mean(np.linalg.norm(reproj - pixels_corriges, axis=1)))

    resultat = {
        "H": H_image_vers_mur,
        "rvec": rvec,
        "tvec": tvec,
        "erreur_px": erreur,
    }
    return resultat, coins_detectes, presents


def projeter(points_3d, rvec, tvec, K, dist):
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    pixels, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    return np.rint(pixels.reshape(-1, 2)).astype(int)


def projeter_plan(points_2d, H_image_vers_mur, K, dist):
    """Projette des (X,Y) du mur sur le flux couleur, distorsion comprise."""
    points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 1, 2)
    H_mur_vers_image = np.linalg.inv(H_image_vers_mur)
    pixels_corriges = cv2.perspectiveTransform(points_2d, H_mur_vers_image)
    # Inverser la correction de distorsion pour dessiner dans l'image brute.
    p = pixels_corriges.reshape(-1, 2)
    rayons = np.column_stack([
        (p[:, 0] - K[0, 2]) / K[0, 0],
        (p[:, 1] - K[1, 2]) / K[1, 1],
        np.ones(len(p)),
    ])
    resultat, _ = cv2.projectPoints(
        rayons, np.zeros(3), np.zeros(3), K, dist
    )
    return np.rint(resultat.reshape(-1, 2)).astype(int)


def dessiner_repere(image, resultat, K, dist, grille=False):
    rvec, tvec = resultat["rvec"], resultat["tvec"]
    H = resultat["H"]

    # Origine, X et Y calcules par l'homographie du plan du mur.
    # Z seul est projete grace a la pose 3D estimee par solvePnP.
    O, X, Y = [tuple(p) for p in projeter_plan([
        [0, 0], [250, 0], [0, 250]
    ], H, K, dist)]
    Z = tuple(projeter([[0, 0, 180]], rvec, tvec, K, dist)[0])

    # Contour physique du panneau, et NON centres des ArUco.
    contour = projeter_plan([
        [0, 0],
        [LARGEUR_MUR_MM, 0],
        [LARGEUR_MUR_MM, HAUTEUR_MUR_MM],
        [0, HAUTEUR_MUR_MM],
    ], H, K, dist)
    cv2.polylines(image, [contour], True, (0, 255, 255), 2)

    if grille:
        for x in np.arange(0, LARGEUR_MUR_MM + 0.01, 100):
            p, q = projeter_plan([[x, 0], [x, HAUTEUR_MUR_MM]],
                                H, K, dist)
            cv2.line(image, tuple(p), tuple(q), (155, 155, 155), 1)
        for y in np.arange(0, HAUTEUR_MUR_MM + 0.01, 100):
            p, q = projeter_plan([[0, y], [LARGEUR_MUR_MM, y]],
                                H, K, dist)
            cv2.line(image, tuple(p), tuple(q), (155, 155, 155), 1)

    cv2.arrowedLine(image, O, X, (0, 0, 255), 3, tipLength=0.15)
    cv2.arrowedLine(image, O, Y, (0, 255, 0), 3, tipLength=0.15)
    cv2.arrowedLine(image, O, Z, (255, 0, 0), 3, tipLength=0.15)
    for nom, p, couleur in (("X", X, (0, 0, 255)),
                           ("Y", Y, (0, 255, 0)),
                           ("Z", Z, (255, 0, 0)),
                           ("O", O, (0, 255, 255))):
        cv2.putText(image, nom, (p[0] + 5, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, couleur, 2)


def enregistrer(resultat, K, dist, intr):
    donnees = {
        "largeur_mur_mm": LARGEUR_MUR_MM,
        "hauteur_mur_mm": HAUTEUR_MUR_MM,
        "resolution": [LARGEUR_IMAGE, HAUTEUR_IMAGE],
        "camera_matrix": K.tolist(),
        "distortion": dist.flatten().tolist(),
        "H_pixel_undistordu_vers_mur": resultat["H"].tolist(),
        "rvec": resultat["rvec"].flatten().tolist(),
        "tvec": resultat["tvec"].flatten().tolist(),
        "erreur_reprojection_px": resultat["erreur_px"],
        "camera_distortion_model_original": str(intr.model),
        "camera_distortion_coefficients_original": list(intr.coeffs),
        "coordonnees_image": (
            "image_rectifiee" if intr.model == rs.distortion.inverse_brown_conrady
            else "image_brute"
        ),
    }
    FICHIER_CALIBRATION.write_text(json.dumps(donnees, indent=2), encoding="utf-8")
    print("Calibration sauvegardee :", FICHIER_CALIBRATION)


