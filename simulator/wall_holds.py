"""Interface simple pour tes collègues : image(s) du mur -> positions des prises (col, row).

Dans leur code :

    import cv2
    from wall_holds import detect_holds, save_holds, load_holds

    holds = detect_holds([cv2.imread("cam_left.png"), cv2.imread("cam_right.png")])   # 1 ou 2 images
    save_holds(holds, "holds.json")
    # ... ou directement : [(h["col"], h["row"]) for h in holds]

Chaque prise : {"col", "row", "x_mm", "y_mm", "color", "n_cams"}
  col 0..51, row 0..92 : même convention que sim.wall.holds du simulateur (h.col, h.row) ;
  x_mm, y_mm : position du centre de la prise sur le mur, origine en bas à gauche de la zone texturée, y vers le haut.

En ligne de commande :

    python wall_holds.py cam_left.png cam_right.png -o holds.json       # sur des images (RGB seul, sans marqueur)
    uv run --with opencv-python python wall_holds.py --sim N2 -o holds_N2.json      # dans le simulateur + vérification
    uv run --with opencv-python python wall_holds.py --sim N3 --move 0.1 --seed 4    # caméras déplacées
"""
import json
import sys

import cv2
import numpy as np

import perception_rgb as P


def detect_holds(images, colors=None, K=None, verbose=True):
    """images : liste d'images BGR (OpenCV) du MEME mur, vues par des caméras différentes. Retourne la liste des prises.

    Les images où le panneau n'est pas entièrement visible / reconnu sont ignorées (le statut est affiché).
    Les prises vues par plusieurs caméras sont fusionnées. `colors` : plages HSV (défaut : couleurs du simulateur
    si tu passes P.SIM_COLORS, sinon orange/vert/bleu du vrai mur).
    """
    found = []
    for i, img in enumerate(images):
        res = P.analyze(img, colors, K=K)
        if verbose:
            print(f"image {i} ({img.shape[1]}x{img.shape[0]}) : {res['status']}")
        if res["status"].startswith("OK"):
            found += [dict(h, cam=i) for h in res["holds"]]
    merged = []
    for h in found:
        for m in merged:
            if np.hypot(m["x"] - h["x"], m["y"] - h["y"]) < 30.0:        # même prise vue par 2 caméras
                n = m["n"]
                m["x"], m["y"] = (m["x"] * n + h["x"]) / (n + 1), (m["y"] * n + h["y"]) / (n + 1)
                m["n"] = n + 1
                break
        else:
            merged.append(dict(h, n=1))
    out = []
    for m in merged:
        col, row = P.mm_to_hole_index(m["x"], m["y"], m.get("offset_mm"))
        out.append({"col": int(col), "row": int(row), "x_mm": round(float(m["x"]), 1), "y_mm": round(float(m["y"]), 1),
                    "color": m["name"], "n_cams": m["n"]})
    return sorted(out, key=lambda h: (h["row"], h["col"]))


def save_holds(holds, path):
    with open(path, "w") as f:
        json.dump({"grid": {"cols": P.N_HOLE_COLS, "rows": P.N_HOLE_ROWS, "pitch_mm": P.HOLE_SPACING_MM}, "holds": holds}, f, indent=1)


def to_cells(holds):
    """Ensemble de (col, row) : le format le plus simple pour un planificateur.  (18, 2) in cells -> prise là ?"""
    return {(h["col"], h["row"]) for h in holds}


def load_holds(path):
    with open(path) as f:
        return json.load(f)["holds"]


def compare(holds, truth_holds):
    """Compare aux vraies prises du simulateur. Retourne (exactes, fausses, manquées) en ensembles de (col, row)."""
    got = {(h["col"], h["row"]) for h in holds}
    truth = {(h.col, h.row) for h in truth_holds}
    return got & truth, got - truth, truth - got


def _sim_main(argv):
    from sim_perception import make_sim, render_bgr
    wall = argv[argv.index("--sim") + 1]
    move = float(argv[argv.index("--move") + 1]) if "--move" in argv else 0.0
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 0
    sim = make_sim(wall, True, (1920, 1080), markers="--markers" in argv, move=move, seed=seed)
    imgs = [render_bgr(sim, n) for n in ("cam_left", "cam_right")]
    holds = detect_holds(imgs, P.SIM_COLORS)
    ok, fp, miss = compare(holds, sim.wall.holds)
    print(f"\nMur {wall} : {len(sim.wall.holds)} prises réelles, {len(holds)} détectées : "
          f"{len(ok)} exactes, {len(fp)} fausses {sorted(fp)}, {len(miss)} manquées {sorted(miss)}")
    return holds, argv


def main():
    argv = sys.argv[1:]
    out = argv[argv.index("-o") + 1] if "-o" in argv else "holds.json"
    if "--sim" in argv:
        holds, _ = _sim_main(argv)
    else:
        paths = [a for i, a in enumerate(argv) if not a.startswith("-") and (i == 0 or argv[i - 1] != "-o")]
        imgs = [cv2.imread(p) for p in paths]
        if not imgs or any(im is None for im in imgs):
            sys.exit("usage : python wall_holds.py image1.png [image2.png] -o holds.json   |   --sim N2")
        holds = detect_holds(imgs)
        print(f"{len(holds)} prises")
    save_holds(holds, out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()