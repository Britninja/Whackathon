"""Détection des prises d'escalade en RGB seul (pas d'ArUco, pas de profondeur) : enchaînement des étapes.

Le détail de chaque étape est dans le dossier perception/ :
  1. board_region.py   où est le panneau ? (la zone où les trous forment une grille régulière dense)
  2. board_corners.py  ses 4 coins, recalés au sous-pixel
  3. hole_grid.py      numéroter les trous -> homographie précise -> vue de face (geometry.make_front_view)
  4. holds.py          les prises par couleur dans la vue de face (réglages des couleurs : colors.py)
  +  camera_pose.py    position de la caméra par rapport au mur

Usage
  python perception_rgb.py              # caméra RealSense (couleur seule)
  python perception_rgb.py photo.png     # test hors-ligne sur une image
Touches : q quitter, s afficher le tableau des prises, clic gauche dans "Camera" : HSV du pixel.

Les autres scripts font `import perception_rgb as P` puis P.analyze, P.draw, P.SIM_COLORS, P.mm_to_hole_index... :
ces noms sont ré-exportés ci-dessous.
"""
import sys

import cv2
import numpy as np

# ---- noms ré-exportés pour les autres scripts (wall_holds, wall_map, sim_perception, live_cameras...)
from perception.board_corners import (corners_inside_image, fit_board_quad, homography_from_corners,  # noqa: F401
                                      quad_is_plausible, sides_cut_by_image_border)
from perception.board_region import MAX_IMAGE_SIDE_PX, best_board_mask, find_candidate_grids  # noqa: F401
from perception.camera_pose import approx_realsense_intrinsics, board_mm_to_wall_m, camera_pose_from_points  # noqa: F401
from perception.colors import COLORS, SIM_COLORS  # noqa: F401
from perception.display import draw, print_table  # noqa: F401
from perception.geometry import (BOARD_H_MM, BOARD_ORIGIN_X_MM, BOARD_ORIGIN_Y_MM, BOARD_W_MM,  # noqa: F401
                                 HOLE_SPACING_MM, N_HOLE_COLS, N_HOLE_ROWS, PX_PER_MM, SIM_HOLD_OFFSET_MM,
                                 board_corners_px, make_front_view, mm_to_hole_index)
from perception.hole_grid import fit_grid_homography
from perception.hole_grid_fallback import fit_grid_homography_fallback
from perception.holds import detect_holds  # noqa: F401


def analyze(image, colors=None, K=None):
    """Tout le pipeline sur une image BGR. Ne lève jamais d'exception : renvoie un dict.

    Clés du résultat :
      status    texte ("OK : ..." si tout va bien, sinon la raison de l'échec)
      mask      masque du panneau (image binaire)
      quad      4 coins du panneau en pixels (HG, HD, BD, BG)
      inside    pour chaque coin : est-il dans l'image ?
      rect      vue de face du panneau
      holds     liste des prises (voir perception/holds.py)
      H_px2mm   homographie pixel caméra -> mm du panneau
      H_px2rect homographie pixel caméra -> pixel de la vue de face
      pose      pose de la caméra (voir perception/camera_pose.py), seulement si la grille est reconnue
      lattice   nombre de trous utilisés pour l'homographie (None si la grille n'est pas reconnue)

    Une photo de plus de 1920 px (téléphone) est réduite pour trouver le panneau et la grille, puis la vue de face et
    les prises sont refaites en pleine résolution."""
    biggest_side = max(image.shape[:2])
    if biggest_side <= 1.15 * MAX_IMAGE_SIDE_PX:
        return analyze_image(image, colors, K)
    scale = MAX_IMAGE_SIDE_PX / biggest_side
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    K_small = None
    if K is not None:
        K_small = np.asarray(K, np.float64).copy()
        K_small[:2] *= scale
    result = analyze_image(small, colors, K_small)
    if result["quad"] is not None:
        result["quad"] = result["quad"] / scale
    if result["mask"] is not None:
        result["mask"] = cv2.resize(result["mask"], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if result["H_px2mm"] is not None:
        H_full = result["H_px2mm"] @ np.diag([scale, scale, 1.0])      # pixel pleine résolution -> mm
        result["H_px2mm"] = H_full
        result["rect"], result["H_px2rect"] = make_front_view(image, H_full)
        if result["status"].startswith("OK"):
            result["holds"] = detect_holds(result["rect"], colors)
    return result


def analyze_image(image, colors=None, K=None):
    """Le pipeline sur une image de taille raisonnable (<= ~1920 px)."""
    result = {"status": "", "mask": None, "quad": None, "inside": None, "rect": None,
              "holds": [], "H_px2rect": None, "H_px2mm": None, "pose": None, "lattice": None}
    grids = find_candidate_grids(image)

    grid_fit = None
    for permissive in (False, True):         # 2e essai avec un masque plus permissif (vue très oblique)
        # Etape 1 : le panneau
        grid, mask, contour, density, threshold = best_board_mask(image, grids, permissive)
        if mask is None:
            result["status"] = "PANNEAU NON TROUVE"
            return result
        result["mask"] = mask

        # Etape 2 : ses 4 coins, et les contrôles
        quad = fit_board_quad(contour, density, threshold)
        if quad is None:
            result["status"] = "QUADRILATERE NON TROUVE"
            return result
        result["quad"] = quad
        result["inside"] = corners_inside_image(quad, image.shape)
        cut_sides = sides_cut_by_image_border(mask)
        if cut_sides or not all(result["inside"]):
            what = ("bords coupés : " + ", ".join(cut_sides)) if cut_sides else "coins hors image"
            result["status"] = f"PANNEAU INCOMPLET ({what}) - recule la camera"
            return result
        if not quad_is_plausible(quad):
            result["status"] = "CADRE INVALIDE (angles ou proportions incoherents : image trop petite ? passe en 1920x1080)"
            if permissive:
                return result
            continue

        # Etape 3 : homographie précise par la numérotation des trous (méthode de secours si elle échoue)
        H = homography_from_corners(quad)
        px_per_mm = float(np.sqrt(abs(cv2.contourArea(quad.astype(np.float32))) / (BOARD_W_MM * BOARD_H_MM)))
        grid_fit = fit_grid_homography(H, grid[0])
        if grid_fit is None:
            grid_fit = fit_grid_homography_fallback(image, mask, H, px_per_mm)
        if grid_fit is not None:
            break

    if grid_fit is not None:
        H, holes_px, holes_mm, n_holes = grid_fit
        result["lattice"] = n_holes
        result["quad"] = board_corners_px(H)        # coins exacts d'après la grille
    # (sinon on garde H = homographie des 4 coins, moins précise)

    # Etape 4 : vue de face et prises
    result["rect"], result["H_px2rect"] = make_front_view(image, H)
    result["H_px2mm"] = H
    result["holds"] = detect_holds(result["rect"], colors)

    # Pose de la caméra : seulement avec la grille (avec les 4 coins seuls, l'erreur est de ~30 cm)
    if grid_fit is not None:
        if K is None:
            K = approx_realsense_intrinsics(image.shape)
        result["pose"] = camera_pose_from_points(board_mm_to_wall_m(holes_mm), holes_px, K)

    if result["lattice"]:
        result["status"] = f"OK : {len(result['holds'])} prises (grille : {result['lattice']} trous)"
    else:
        result["status"] = f"GRILLE NON RECONNUE : {len(result['holds'])} prises mais positions non fiables, pas de pose"
    return result


def main():
    if len(sys.argv) > 1:                                    # test hors-ligne sur une image
        image = cv2.imread(sys.argv[1])
        result = analyze(image)
        vis, front_vis = draw(image, result)
        print(result["status"])
        print_table(result["holds"])
        cv2.imshow("Camera", vis)
        if front_vis is not None:
            cv2.imshow("Mur de face", front_vis)
        cv2.waitKey(0)
        return

    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    current_hsv = {"image": None}

    def on_mouse(event, x, y, flags, param):            # clic gauche : affiche le HSV du pixel (pour régler les couleurs)
        if event == cv2.EVENT_LBUTTONDOWN and current_hsv["image"] is not None:
            print("HSV:", current_hsv["image"][y, x])
    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", on_mouse)
    last_holds = []
    try:
        while True:
            color_frame = pipeline.wait_for_frames().get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data()).copy()
            current_hsv["image"] = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            result = analyze(image)
            if result["holds"]:
                last_holds = result["holds"]
            vis, front_vis = draw(image, result)
            cv2.imshow("Camera", vis)
            if front_vis is not None:
                cv2.imshow("Mur de face", front_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                print_table(last_holds)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
