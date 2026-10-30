"""Voir quelles prises sont vues ou manquées : une carte du mur + les mêmes prises dessinées sur l'image des caméras.

    from wall_map import save_map, overlay
    save_map(holds, sim.wall.holds, "carte.png")                  # holds = résultat de detect_holds
    cv2.imwrite("cam.png", overlay(img, res, holds, sim.wall.holds))   # res = P.analyze(img, ...)

Vert = prise réelle bien détectée, ROUGE = prise réelle NON vue, jaune = prise détectée qui n'existe pas.
"""
import cv2
import numpy as np

import perception_rgb as P

GREEN, RED, YELLOW = (60, 200, 60), (40, 40, 230), (0, 220, 255)


def _status(holds, truth_holds):
    got = {(h["col"], h["row"]) for h in holds}
    truth = {(h.col, h.row) for h in truth_holds}
    return got & truth, truth - got, got - truth


def save_map(holds, truth_holds, path, cell=12):
    """Carte du mur vue de face (bas du mur en bas) : une case par trou, une barre de 3 trous par prise."""
    ok, missed, false = _status(holds, truth_holds)
    W, H = P.N_COLS * cell, P.N_ROWS * cell
    img = np.full((H + 40, W, 3), 255, np.uint8)
    for c in range(P.N_COLS):
        for r in range(P.N_ROWS):
            cv2.circle(img, (c * cell + cell // 2, H - r * cell - cell // 2), 1, (200, 200, 200), -1)
    for cells, color, thick in ((ok, GREEN, -1), (missed, RED, -1), (false, YELLOW, 2)):
        for c, r in cells:
            cv2.rectangle(img, (c * cell + 2, H - (r + 1) * cell + 2), ((c + 3) * cell - 2, H - r * cell - 2), color, thick)
    cv2.putText(img, f"vert {len(ok)} vues | ROUGE {len(missed)} NON vues | jaune {len(false)} en trop",
                (8, H + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return img


def overlay(img, res, holds, truth_holds):
    """Dessine sur l'image caméra la position des vraies prises (vert vue / rouge manquée) et des prises en trop."""
    out = img.copy()
    if res.get("H_px2mm") is None:
        return out
    ok, missed, false = _status(holds, truth_holds)
    Hinv = np.linalg.inv(res["H_px2mm"])
    for cells, color in ((ok, GREEN), (missed, RED), (false, YELLOW)):
        for c, r in cells:
            x, y = (c + 1.5) * P.HOLE_PITCH, (r + 0.5) * P.HOLE_PITCH + P.HOLD_DY
            u, v = cv2.perspectiveTransform(np.float32([[[x, y]]]), Hinv)[0, 0]
            cv2.circle(out, (int(u), int(v)), 14, color, 3)
    return out