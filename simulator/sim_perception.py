"""Ma perception RGB (perception_rgb.py) sur le simulateur MuJoCo, comparée à la vérité du simulateur.

Lancer depuis simulator/ (là où `uv run` marche), avec perception_rgb.py dans le même dossier :

    uv run python sim_perception.py                  # mur N2, robot suspendu, caméras 1920x1080
    uv run python sim_perception.py N3               # un autre mur
    uv run python sim_perception.py N2 --no-robot    # scène sans robot (plus rapide, aucune prise cachée)
    uv run python sim_perception.py N2 --no-robot --no-markers   # + sans plaques ArUco : le mode "sans marqueurs"
    uv run python sim_perception.py N2 --no-robot --move 0.1     # caméras déplacées au hasard de +-10 cm (--seed 3 : autre tirage)

Ce que ça vérifie, pour chaque caméra, sans aucun ArUco et en RGB seul (comme l'impose le règlement) :
  - le panneau est trouvé, redressé, et ses prises donnent leur (col, row) : comparées à sim.wall.holds ;
  - la POSE de la caméra par rapport au mur (le règlement : « estimer la pose des caméras par rapport au mur est le
    travail du code de l'équipe ») : comparée à la vraie pose du simulateur.

Dans ton code : `holds, cameras = perceive(sim)` rend les prises vues (col, row, x, y en mm) et les caméras estimées
(des `Camera` du simulateur, avec rvec / tvec) ; `ClimbingSim(cameras=cameras)` s'en sert directement.

Pour que ClimbingSim lui-même n'affiche pas les plaques ArUco (mode « sans marqueurs » du règlement), change dans
climbing_sim/sim.py :   build_spec(wall=self.wall, cameras=cameras)   ->   build_spec(wall=self.wall, cameras=cameras, markers=False)
"""
import sys

import cv2
import numpy as np

import perception_rgb as P

CAMERAS = ("cam_left", "cam_right")


def render_bgr(sim, name):
    """Image de la caméra du simulateur au format OpenCV (BGR)."""
    return cv2.cvtColor(sim.render(name), cv2.COLOR_RGB2BGR)


def rotation_error_deg(R1, R2):
    return float(np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1))))


def perceive(sim, cameras=CAMERAS, colors=None, save_prefix=None, verbose=True):
    """Retourne (prises, caméras estimées).

    prises : liste de dicts {col, row, x, y (mm, repère de la zone texturée), n (nombre de caméras qui la voient)} ;
    une prise vue par deux caméras est fusionnée (moyenne).
    caméras estimées : liste de `Camera` du simulateur (rvec / tvec : repère mur -> caméra) pour les caméras dont la
    grille de trous a été reconnue ; les autres sont ignorées et leur statut est affiché.
    """
    from climbing_sim.cameras import Camera
    colors = P.SIM_COLORS if colors is None else colors
    found, estimated = [], []
    for name in cameras:
        cam = sim.cameras[name]
        img = render_bgr(sim, name)
        res = P.analyze(img, colors, K=cam.K())
        if verbose:
            print(f"{name} ({img.shape[1]}x{img.shape[0]}) : {res['status']}")
        if save_prefix:
            vis, rect_vis = P.draw(img, res)
            cv2.imwrite(f"{save_prefix}_{name}.png", vis)
            if rect_vis is not None:
                cv2.imwrite(f"{save_prefix}_{name}_rect.png", rect_vis)
        if not res["status"].startswith("OK"):
            continue
        found += [dict(h, cam=name) for h in res["holds"]]
        if res["pose"] is not None:
            rvec, tvec, err, pos = res["pose"]
            est = Camera(name=name, width=cam.width, height=cam.height, fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
                         rvec=[float(x) for x in rvec], tvec=[float(x) for x in tvec])
            estimated.append(est)
            if verbose:
                dp = np.linalg.norm(pos - cam.position_wall()) * 1000
                da = rotation_error_deg(cam.R_wall_to_cam(), est.R_wall_to_cam())
                print(f"   pose estimée : position ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m, reprojection {err:.2f} px"
                      f"  |  écart à la vraie pose : {dp:.1f} mm, {da:.2f}°")
    merged = []
    for h in found:
        for m in merged:
            if np.hypot(m["x"] - h["x"], m["y"] - h["y"]) < 30.0:         # même prise vue par 2 caméras
                n = m["n"]
                m["x"], m["y"] = (m["x"] * n + h["x"]) / (n + 1), (m["y"] * n + h["y"]) / (n + 1)
                m["n"] = n + 1
                break
        else:
            merged.append(dict(h, n=1))
    for m in merged:
        m["col"], m["row"] = P.mm_to_hole_index(m["x"], m["y"], m.get("offset_mm"))
    return merged, estimated


def evaluate(detected, wall_holds):
    """Compare aux prises du mur. Retourne (stats, dx, dy conseillés pour SIM_HOLD_OFFSET_MM)."""
    truth = [(h.col, h.row) for h in wall_holds]
    # centre attendu de chaque prise dans le repère de la zone texturée (sans décalage de calibration)
    exp = {t: ((t[0] + 1.5) * P.HOLE_SPACING_MM, (t[1] + 0.5) * P.HOLE_SPACING_MM) for t in truth}
    res_dx, res_dy = [], []
    for d in detected:
        t = min(truth, key=lambda t: np.hypot(d["x"] - exp[t][0], d["y"] - exp[t][1]))
        if np.hypot(d["x"] - exp[t][0], d["y"] - exp[t][1]) < 25.0:
            res_dx.append(d["x"] - exp[t][0])
            res_dy.append(d["y"] - exp[t][1])
    dx, dy = (float(np.median(res_dx)), float(np.median(res_dy))) if res_dx else (0.0, 0.0)

    def score(dxx, dyy):
        got = {P.mm_to_hole_index(d["x"], d["y"], (dxx, dyy)) for d in detected}
        return len(got & set(truth)), len(got - set(truth)), len(set(truth) - got)

    stats = {"vrai": len(truth), "detectees": len(detected),
             "avant": score(*P.SIM_HOLD_OFFSET_MM), "apres": score(dx, dy),
             "ecart_mm": float(np.mean(np.hypot(np.array(res_dx) - dx, np.array(res_dy) - dy))) if res_dx else float("nan")}
    return stats, dx, dy


def make_sim(wall, no_robot, size, markers=True, move=0.0, seed=0):
    from climbing_sim import Camera, default_cameras
    from climbing_sim.cameras import d435i_intrinsics
    cams = default_cameras(*size)
    if move > 0:                                   # les organisateurs déplacent légèrement les caméras
        rng = np.random.default_rng(seed)
        kw = d435i_intrinsics(*size)
        moved = []
        for c in cams:
            p = c.position_wall() + rng.uniform(-move, move, 3)
            t = np.array([0.4, 0.7, 0.0]) + rng.uniform(-move, move, 3) * [1, 1, 0]
            moved.append(Camera.looking_at(c.name, p.tolist(), t.tolist(), **kw))
        cams = moved
    if not no_robot:
        from climbing_sim import ClimbingSim
        sim = ClimbingSim(wall=wall, cameras=cams)
        sim.reset_hanging()
        return sim
    # scène sans le robot : même mur, mêmes prises, mêmes caméras (et marqueurs si markers=True)
    import mujoco
    from climbing_sim import Wall, build_spec

    class NoRobot:
        pass
    w = Wall.load(wall)
    spec = build_spec(w, cams, robot=False, markers=markers)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = size
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, size[1], size[0])
    sim = NoRobot()
    sim.wall, sim.cameras = w, {c.name: c for c in cams}

    def render(name):
        renderer.update_scene(data, camera=name)
        return renderer.render()
    sim.render = render
    return sim


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--") and not a.replace(".", "").isdigit()]
    wall = args[0] if args else "N2"
    size = (1280, 720) if "--720" in sys.argv else (1920, 1080)
    move = float(sys.argv[sys.argv.index("--move") + 1]) if "--move" in sys.argv else 0.0
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    sim = make_sim(wall, "--no-robot" in sys.argv, size, markers="--no-markers" not in sys.argv, move=move, seed=seed)
    holds, cams = perceive(sim, save_prefix=f"perception_{wall}")
    stats, dx, dy = evaluate(holds, sim.wall.holds)
    ok0, fp0, miss0 = stats["avant"]
    ok1, fp1, miss1 = stats["apres"]
    print(f"\nMur {wall} : {stats['vrai']} prises dans le JSON, {stats['detectees']} détectées")
    sim_dx, sim_dy = P.SIM_HOLD_OFFSET_MM
    print(f"  réglages actuels  (SIM_HOLD_OFFSET_MM=({sim_dx:.1f}, {sim_dy:.1f})) : "
          f"{ok0} exactes, {fp0} fausses, {miss0} manquées")
    print(f"  après calibration (SIM_HOLD_OFFSET_MM=({dx:.1f}, {dy:.1f})) : "
          f"{ok1} exactes, {fp1} fausses, {miss1} manquées, écart résiduel moyen {stats['ecart_mm']:.1f} mm")
    print(f"=> mets SIM_HOLD_OFFSET_MM = ({dx:.1f}, {dy:.1f}) dans perception/geometry.py")
    if cams:
        from climbing_sim import save_cameras
        save_cameras(cams, f"cameras_estimees_{wall}.json")
        print(f"{len(cams)} caméra(s) estimée(s) enregistrée(s) dans cameras_estimees_{wall}.json "
              f"(à ouvrir avec climbing-sim-cameras, ou ClimbingSim(cameras='cameras_estimees_{wall}.json'))")


if __name__ == "__main__":
    main()