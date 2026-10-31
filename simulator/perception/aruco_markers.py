"""Etapes 1+2 avec marqueurs : le panneau localisé par ses 4 marqueurs ArUco de coin (remplace board_region et
board_corners quand les marqueurs sont visibles).

Géométrie (climbing_sim/aruco.py, conforme au règlement) : DICT_4X4_50, IDs 1 = haut-gauche, 2 = haut-droit,
3 = bas-gauche, 4 = bas-droit ; code noir de 100 mm, bordure blanche de 18 mm, plaque de 136 mm dont le coin
EXTERIEUR est posé sur le coin du panneau de 800 x 1400 mm.

Fonctions principales : detect_markers(image) puis aruco_homography(found).
"""
import cv2
import numpy as np

from perception.geometry import BOARD_ORIGIN_X_MM, BOARD_ORIGIN_Y_MM

WALL_W, WALL_H = 800.0, 1400.0      # le panneau entier (mm), repère mur, origine en bas à gauche
CODE, BORDER = 100.0, 18.0          # côté du code noir, bordure blanche de la plaque
PLATE = CODE + 2 * BORDER           # 136 mm
IDS = (1, 2, 3, 4)                  # 1 haut-gauche, 2 haut-droit, 3 bas-gauche, 4 bas-droit

# 2e passe, seulement quand un marqueur manque : seuillage plus fin, contour moins strict, bordure abîmée tolérée
# (un câble, une ombre ou un reflet en travers d'un marqueur suffisent à le faire rejeter). Elle sort aussi des faux
# marqueurs : on ne garde les IDs 1 à 4 qu'après vérification géométrique (voir detect_markers).
LOOSE = {"adaptiveThreshWinSizeMin": 3, "adaptiveThreshWinSizeMax": 53, "adaptiveThreshWinSizeStep": 5,
         "polygonalApproxAccuracyRate": 0.08, "minMarkerPerimeterRate": 0.01, "maxErroneousBitsInBorderRate": 0.6,
         "errorCorrectionRate": 1.0, "perspectiveRemoveIgnoredMarginPerCell": 0.33}
MAX_RESIDUAL_MM = 20.0      # écart toléré entre un marqueur et sa place théorique sur le panneau


def marker_corners_mm(marker_id):
    """Les 4 coins du code noir du marqueur, en mm dans le repère de la zone percée (origine en bas à gauche, y vers
    le haut), dans l'ordre d'OpenCV : HG, HD, BD, BG."""
    x0 = 0.0 if marker_id in (1, 3) else WALL_W - PLATE               # coin de la plaque, repère mur
    y0 = 0.0 if marker_id in (3, 4) else WALL_H - PLATE
    x0, y0 = x0 + BORDER - BOARD_ORIGIN_X_MM, y0 + BORDER - BOARD_ORIGIN_Y_MM   # -> coin bas-gauche du code
    return np.float32([[x0, y0 + CODE], [x0 + CODE, y0 + CODE], [x0 + CODE, y0], [x0, y0]])


# ---------------------------------------------------------------------- détection OpenCV (compatible anciennes versions)

_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        get = getattr(cv2.aruco, "getPredefinedDictionary", None) or cv2.aruco.Dictionary_get
        dictionary = get(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "ArucoDetector"):                 # OpenCV >= 4.7
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # coins au sous-pixel : la pose en dépend
            _DETECTOR = cv2.aruco.ArucoDetector(dictionary, params)
        else:                                                   # anciennes versions
            params = cv2.aruco.DetectorParameters_create()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            _DETECTOR = (dictionary, params)
    return _DETECTOR


def _detect(image, loose=False):
    """{id: coins (4, 2) en pixels} des marqueurs 1 à 4 vus dans l'image (un seul par id : le premier)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if loose:
        detector = _detector()
        dictionary = detector[0] if isinstance(detector, tuple) else detector.getDictionary()
        new_api = hasattr(cv2.aruco, "ArucoDetector")
        params = cv2.aruco.DetectorParameters() if new_api else cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        for key, value in LOOSE.items():
            setattr(params, key, value)
        detector = cv2.aruco.ArucoDetector(dictionary, params) if new_api else (dictionary, params)
    else:
        detector = _detector()
    if isinstance(detector, tuple):
        corners, ids, _ = cv2.aruco.detectMarkers(gray, detector[0], parameters=detector[1])
    else:
        corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    found = {}
    for c, i in zip(corners, ids.ravel()):
        if int(i) in IDS:
            found.setdefault(int(i), c.reshape(4, 2).astype(np.float32))
    return found


# ---------------------------------------------------------------------- API

def aruco_homography(found):
    """Homographie pixel -> mm (repère de la zone percée) par les marqueurs trouvés (4 points chacun). None si < 2."""
    if len(found) < 2:
        return None
    ids = sorted(found)
    src = np.concatenate([found[m] for m in ids])
    dst = np.concatenate([marker_corners_mm(m) for m in ids])
    H, _ = cv2.findHomography(src, dst, 0)                      # moindres carrés sur 8 à 16 points
    return H


def residual_mm(found, marker_id, H):
    """Écart maximal, en mm sur le panneau, entre les coins vus du marqueur et sa place théorique."""
    mm = cv2.perspectiveTransform(found[marker_id].reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.abs(mm - marker_corners_mm(marker_id)).max())


def detect_markers(image):
    """{id: coins (4, 2) en pixels} pour les marqueurs 1 à 4 du panneau. Les autres IDs (robot...) sont ignorés.

    Si les 4 ne sont pas vus, une 2e passe plus permissive essaie de récupérer les manquants, et n'accepte un marqueur
    que s'il tombe à sa place sur le panneau d'après les marqueurs déjà sûrs (un faux marqueur est donc rejeté)."""
    found = _detect(image)
    if len(found) == len(IDS) or len(found) < 2:
        return found
    H = aruco_homography(found)
    if H is None or not np.isfinite(H).all():
        return found
    for marker_id, corners in _detect(image, loose=True).items():
        if marker_id in found:
            continue
        found[marker_id] = corners
        if residual_mm(found, marker_id, H) > MAX_RESIDUAL_MM:       # faux marqueur ou mauvais coins : on le jette
            del found[marker_id]
    return found
