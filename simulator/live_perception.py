"""Perception en direct dans le simulateur, robot suspendu au mur (non testé de mon côté : le modèle du robot n'est pas
téléchargeable chez moi).

    uv run --with opencv-python python live_perception.py N2
    uv run --with opencv-python python live_perception.py random_walls/r100_3.json --view     # + fenêtre MuJoCo

Toutes les 100 étapes de simulation : rend les 2 caméras, détecte les prises, compare à la vérité du simulateur et écrit
live_<caméra>.png (prises dessinées : vert = vue, ROUGE = manquée) et live_carte.png (carte du mur, mêmes couleurs). Ouvre ces PNG dans VS Code ; ils se mettent à jour tout seuls.
"""
import sys

import cv2

import perception_rgb as P
from climbing_sim import ClimbingSim, default_cameras
from sim_perception import render_bgr
from wall_holds import compare, detect_holds
from wall_map import overlay, save_map

wall = next((a for a in sys.argv[1:] if not a.startswith("--")), "N2")
sim = ClimbingSim(wall=wall, cameras=default_cameras(1920, 1080))          # 1080p obligatoire
obs = sim.reset_hanging()
if "--view" in sys.argv:
    sim.view()

for t in range(2001):
    obs = sim.step(obs["q"])                  # le robot reste en position
    if t % 100:
        continue
    imgs = [render_bgr(sim, n) for n in ("cam_left", "cam_right")]
    holds = detect_holds(imgs, P.SIM_COLORS, verbose=False)
    ok, fp, miss = compare(holds, sim.wall.holds)
    print(f"t={t:4d} : {len(holds)} prises détectées, {len(ok)}/{len(sim.wall.holds)} exactes, "
          f"{len(fp)} fausses, {len(miss)} manquées {sorted(miss)[:6]}")
    for name, img in zip(("cam_left", "cam_right"), imgs):
        res = P.analyze(img, P.SIM_COLORS)
        cv2.imwrite(f"raw_{name}.png", img)                    # image brute, sans dessin : pour tester wall_holds.py dessus
        vis, _ = P.draw(img, res)
        cv2.imwrite(f"live_{name}.png", overlay(vis, res, holds, sim.wall.holds))
    save_map(holds, sim.wall.holds, "live_carte.png")