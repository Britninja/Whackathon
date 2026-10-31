"""Version ArUco de la perception : le panneau est localisé par les 4 marqueurs des coins au lieu de la grille de trous.

C'est un REMPLACANT DIRECT de perception_rgb : même `analyze(frame, colors, K)`, même dict de résultats, même `draw`.
Pour l'utiliser partout (wall_holds.py, live_cameras.py, live_perception.py), remplace la ligne

    import perception_rgb as P          par          import perception_aruco as P

Seule l'étape « trouver le panneau » change (perception/aruco_markers.py au lieu de board_region + board_corners) ;
la suite est la même que perception_rgb (grille de trous, vue de face, prises par couleur, pose). On y gagne la
robustesse (plus de panneau voisin qui gagne, plus de vue oblique ambiguë, le robot peut cacher la moitié du mur) et la
vitesse ; on y perd le facteur x1,25 du règlement, et il faut que les marqueurs soient visibles et nets.

Par défaut `refine=True` : les marqueurs donnent l'homographie de départ, puis la grille de trous l'affine (des milliers
de points au lieu de 16). C'est le meilleur des deux. `refine=False` ne garde que les marqueurs.

Mesuré en 1920x1080 dans le simulateur (N1, N2, N3 et les murs aléatoires, caméras déplacées de +-10 cm) :

    refine=True    ~480 ms/image   100 % de prises exactes   pose 0,4 a 0,9 mm et 0,02 deg
    refine=False    ~70 ms/image   100 % de prises exactes   pose ~4 mm
    detection des marqueurs seule : ~14 ms

Mêmes scores que la version sans marqueur, mais sans sa fragilité : sur une photo de téléphone prise de biais, avec un
panneau voisin et le robot accroché au mur, la version sans marqueur se trompait de panneau, celle-ci trouve le bon du
premier coup (résidus de 0,8 à 1,9 mm sur les marqueurs). Inversement, sur une image où aucun marqueur n'est visible
(cadrage serré), cette version ne peut rien et c'est perception_rgb qu'il faut : les deux sont complémentaires.

En ligne de commande :

    uv run --with opencv-python python perception_aruco.py photo.jpg          # sur une image : écrit aruco_*.png
    uv run --with opencv-python python perception_aruco.py --sim N2           # dans le simulateur (marqueurs dessinés) + vérification
    uv run --with opencv-python python perception_aruco.py --sim N2 --move 0.1 --seed 3 --no-refine
"""
import sys

import cv2
import numpy as np

# ---- mêmes noms que perception_rgb, pour pouvoir remplacer l'un par l'autre (wall_holds, wall_map, sim_perception...)
from perception.aruco_markers import IDS, aruco_homography, detect_markers, marker_corners_mm  # noqa: F401
from perception.board_corners import corners_inside_image
from perception.camera_pose import approx_realsense_intrinsics, board_mm_to_wall_m, camera_pose_from_points  # noqa: F401
from perception.colors import COLORS, SIM_COLORS  # noqa: F401
from perception.display import draw, print_table  # noqa: F401
from perception.geometry import (BOARD_H_MM, BOARD_ORIGIN_X_MM, BOARD_ORIGIN_Y_MM, BOARD_W_MM,  # noqa: F401
                                 HOLE_SPACING_MM, N_HOLE_COLS, N_HOLE_ROWS, SIM_HOLD_OFFSET_MM,
                                 board_corners_px, make_front_view, mm_to_hole_index)
from perception.hole_grid import fit_grid_homography_from_markers
from perception.holds import detect_holds

MIN_MARKERS = 3                     # 3 marqueurs suffisent à une homographie saine ; avec 2 la géométrie est trop plate


def analyze(frame, colors=None, K=None, refine=True, min_markers=MIN_MARKERS):
    """Même contrat que perception_rgb.analyze : ne lève jamais, renvoie le dict de résultats + statut.
    `res["markers"]` en plus : la liste des IDs vus."""
    res = {"status": "", "mask": None, "quad": None, "inside": None, "rect": None, "holds": [],
           "H_px2rect": None, "H_px2mm": None, "pose": None, "lattice": None, "markers": []}
    if frame is None or frame.size == 0:
        res["status"] = "IMAGE VIDE"
        return res

    # Etapes 1+2 : le panneau par ses marqueurs
    found = detect_markers(frame)
    res["markers"] = sorted(found)
    if len(found) < max(2, min_markers):
        res["status"] = (f"ARUCO : {len(found)} marqueur(s) vu(s) {sorted(found)}, il en faut {min_markers} "
                         f"(manquants : {[m for m in IDS if m not in found]}) - recadre, rapproche ou eclaire")
        return res
    H = aruco_homography(found)
    if H is None or not np.isfinite(H).all():
        res["status"] = "ARUCO : homographie impossible (marqueurs mal detectes)"
        return res

    # Etape 3 : affinage par la grille de trous
    holes_px = holes_mm = None
    n_holes = 0
    if refine:
        grid_fit = fit_grid_homography_from_markers(frame, H)
        if grid_fit is not None:
            H, holes_px, holes_mm, n_holes = grid_fit
    res["quad"] = board_corners_px(H)
    res["inside"] = corners_inside_image(res["quad"], frame.shape)
    res["lattice"] = n_holes or None

    # Etape 4 : vue de face et prises
    rect, H_rect = make_front_view(frame, H)
    res["rect"], res["H_px2mm"], res["H_px2rect"] = rect, H, H_rect
    res["holds"] = detect_holds(rect, colors)

    # Pose de la caméra
    if K is None:
        K = approx_realsense_intrinsics(frame.shape)
    if n_holes:                     # par la grille : des milliers de points, erreur submillimétrique
        res["pose"] = camera_pose_from_points(board_mm_to_wall_m(holes_mm), holes_px, K)
    else:                           # par les seuls coins des marqueurs (8 à 16 points)
        ids = sorted(found)
        corners_mm = np.concatenate([marker_corners_mm(m) for m in ids])
        res["pose"] = camera_pose_from_points(board_mm_to_wall_m(corners_mm), np.concatenate([found[m] for m in ids]), K)

    hors = [n for n, ok in zip(("HG", "HD", "BD", "BG"), res["inside"]) if not ok]
    coin = f", coins hors image : {', '.join(hors)}" if hors else ""
    grille = f" + grille : {n_holes} trous" if n_holes else " seuls (grille non reconnue)"
    res["status"] = f"OK : {len(res['holds'])} prises (ArUco : {len(found)} marqueurs{grille}{coin})"
    return res


def _sim(argv):
    """Test dans le simulateur, marqueurs dessinés sur le mur."""
    import sim_perception as SP
    import wall_holds as WH
    WH.P = sys.modules[__name__]                 # wall_holds appelle P.analyze : on lui donne CE module
    wall = argv[argv.index("--sim") + 1]
    move = float(argv[argv.index("--move") + 1]) if "--move" in argv else 0.0
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 0
    sim = SP.make_sim(wall, True, (1920, 1080), markers=True, move=move, seed=seed)
    imgs = [SP.render_bgr(sim, n) for n in ("cam_left", "cam_right")]
    holds = WH.detect_holds(imgs, SIM_COLORS, verbose=True)
    ok, fp, miss = WH.compare(holds, sim.wall.holds)
    print(f"\nMur {wall} : {len(sim.wall.holds)} prises réelles, {len(holds)} détectées : "
          f"{len(ok)} exactes, {len(fp)} fausses {sorted(fp)}, {len(miss)} manquées {sorted(miss)}")
    WH.save_holds(holds, "holds_aruco.json")
    print("-> holds_aruco.json")


def main():
    argv = sys.argv[1:]
    refine = "--no-refine" not in argv
    if "--sim" in argv:
        _sim(argv)
        return
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        sys.exit(__doc__)
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            sys.exit(f"image illisible : {path}")
        res = analyze(frame, refine=refine)
        print(f"{path} ({frame.shape[1]}x{frame.shape[0]}) : {res['status']}")
        if res["pose"] is not None:
            print(f"   camera / mur : {np.round(res['pose'][3], 3)} m, erreur {res['pose'][2]:.2f} px")
        print_table(res["holds"])
        vis, rect_vis = draw(frame, res)
        for m, c in detect_markers(frame).items():              # les marqueurs vus, en magenta
            cv2.polylines(vis, [c.astype(int)], True, (255, 0, 255), 3)
            cv2.putText(vis, str(m), tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
        cv2.imwrite("aruco_vue.png", vis)
        if rect_vis is not None:
            cv2.imwrite("aruco_mur_de_face.png", rect_vis)
        print("-> aruco_vue.png, aruco_mur_de_face.png")


if __name__ == "__main__":
    main()
