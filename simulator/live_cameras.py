"""Détection des prises en direct sur les VRAIES caméras (RGB seul, sans marqueur), avec une fenêtre par caméra.

    uv run --with opencv-python --with pyrealsense2 python live_cameras.py                 # les 2 premières RealSense
    uv run --with opencv-python --with pyrealsense2 python live_cameras.py --serial 1234 5678   # choisir par numéro de série
    uv run --with opencv-python python live_cameras.py --cv 0 2                           # des webcams ordinaires (index OpenCV)
    uv run --with opencv-python python live_cameras.py --images raw_cam_left.png raw_cam_right.png   # sans caméra, sur des images

Options : --K fx fy cx cy (webcams / images : donne la pose), --sim-colors (couleurs du simulateur au lieu de orange / vert / bleu du vrai mur), --once (une passe puis quitte, sans fenêtre de 5 s).

Ce qu'on voit : pour chaque caméra, l'image avec le contour du panneau, les coins HG/HD/BD/BG, et les prises détectées
(col,row) ; le titre dit le statut ("OK : N prises", "PANNEAU INCOMPLET - recule la camera"...). Fenêtre « Mur de face » :
le panneau redressé. L'analyse prend ~0,3 s par image : l'affichage tourne à 2-3 images/s, mais la capture, elle, ne
prend aucun retard.

Une seconde après la première prise détectée, le script collecte pendant 5 s puis ajoute automatiquement à holds.json les prises vues au moins une fois. Les positions déjà présentes sont conservées.
Touches (clique d'abord sur une fenêtre) : q quitter | s sauver raw_cam*.png + ajouter les prises courantes à holds.json |
p afficher le tableau des prises | (RealSense : la position de la caméra par rapport au mur s'affiche, et `s` écrit poses.json) | clic gauche sur l'image : affiche le HSV du pixel (pour régler les couleurs).
"""
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

import perception_rgb as P
from wall_holds import detect_holds, save_holds

W, H = 1920, 1080                 # 1080p obligatoire : en 720p les trous sont trop petits
SHOW_W = 960                      # largeur d'affichage


def arg_list(flag):
    if flag not in sys.argv:
        return []
    out = []
    for a in sys.argv[sys.argv.index(flag) + 1:]:
        if a.startswith("--"):
            break
        out.append(a)
    return out


def cli_K():
    """--K fx fy cx cy : paramètres internes à la main (webcams / images). Sans eux : pas de pose."""
    v = arg_list("--K")
    return np.array([[float(v[0]), 0, float(v[2])], [0, float(v[1]), float(v[3])], [0, 0, 1]]) if len(v) == 4 else None


class Camera:
    """Une caméra : un fil de capture (toujours la dernière image) et un fil d'analyse (la dernière image analysée)."""

    def __init__(self, name, grab, close, colors, K=None):
        self.name, self.grab, self.close, self.colors, self.K = name, grab, close, colors, K
        self.lock = threading.Lock()
        self.frame = None                       # dernière image capturée
        self.shown = None                       # (image analysée, résultat)
        self.running = True
        threading.Thread(target=self._capture, daemon=True).start()
        threading.Thread(target=self._analyse, daemon=True).start()

    def _capture(self):
        while self.running:
            f = self.grab()
            if f is not None:
                with self.lock:
                    self.frame = f

    def _analyse(self):
        last = None
        while self.running:
            with self.lock:
                f = self.frame
            if f is None or f is last:
                time.sleep(0.01)
                continue
            last = f
            res = P.analyze(f, self.colors, K=self.K)
            with self.lock:
                self.shown = (f, res)

    def stop(self):
        self.running = False
        time.sleep(0.2)
        self.close()


def open_realsense(colors):
    import pyrealsense2 as rs
    serials = arg_list("--serial") or [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()][:2]
    if not serials:
        sys.exit("Aucune RealSense trouvée (branche-la, ou utilise --cv / --images).")
    cams = []
    for s in serials:
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(s)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)      # couleur seule : ni profondeur ni IMU
        prof = pipe.start(cfg)
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()      # paramètres internes de CETTE caméra
        K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]])
        cams.append(Camera(f"cam {s}",
                           lambda p=pipe: np.asanyarray(p.wait_for_frames().get_color_frame().get_data()).copy(),
                           pipe.stop, colors, K))
    return cams


def open_opencv(colors):
    cams = []
    for i in arg_list("--cv"):
        cap = cv2.VideoCapture(int(i))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cams.append(Camera(f"webcam {i}", lambda c=cap: c.read()[1], cap.release, colors, cli_K()))
    return cams


def open_images(colors):
    cams = []
    for p in arg_list("--images"):
        img = cv2.imread(p)
        if img is None:
            sys.exit(f"image illisible : {p}")
        cams.append(Camera(p, lambda im=img: (time.sleep(0.05), im.copy())[1], lambda: None, colors, cli_K()))
    return cams


def add_holds_to_file(new_holds, path=None):
    """Ajoute les prises détectées au JSON existant sans retirer de position."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holds.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            holds = json.load(f).get("holds", [])
    else:
        holds = []

    known_cells = {(int(h["col"]), int(h["row"])) for h in holds}
    added = []
    for hold in new_holds:
        cell = (int(hold["col"]), int(hold["row"]))
        if cell not in known_cells:
            holds.append(hold)
            known_cells.add(cell)
            added.append(hold)

    if added or not os.path.exists(path):
        save_holds(holds, path)
    return holds, added


def holds_from_observations(observations):
    """Fusionne les observations faites pendant la capture par case de la grille."""
    by_cell = {}
    for camera_name, hold in observations:
        cell = (int(hold["col"]), int(hold["row"]))
        entry = by_cell.setdefault(cell, {"samples": [], "cameras": set(), "colors": {}})
        entry["samples"].append(hold)
        entry["cameras"].add(camera_name)
        entry["colors"][hold["name"]] = entry["colors"].get(hold["name"], 0) + 1

    merged = []
    for (col, row), entry in sorted(by_cell.items(), key=lambda item: (item[0][1], item[0][0])):
        samples = entry["samples"]
        color = max(entry["colors"], key=entry["colors"].get)
        merged.append({
            "col": col,
            "row": row,
            "x_mm": round(float(np.mean([h["x"] for h in samples])), 1),
            "y_mm": round(float(np.mean([h["y"] for h in samples])), 1),
            "color": color,
            "n_cams": len(entry["cameras"]),
        })
    return merged


def main():
    colors = P.SIM_COLORS if "--sim-colors" in sys.argv else None
    if "--images" in sys.argv:
        cams = open_images(colors)
    elif "--cv" in sys.argv:
        cams = open_opencv(colors)
    else:
        cams = open_realsense(colors)
    once = "--once" in sys.argv
    last_hsv = {}
    capture_seconds = 5.0
    capture_delay = 1.0
    first_detection_at = None
    capture_started = None
    last_recorded = {}
    observations = []

    def on_mouse(ev, x, y, flags, name):
        if ev == cv2.EVENT_LBUTTONDOWN and name in last_hsv:
            k = last_hsv[name].shape[1] / SHOW_W
            print(f"{name} HSV:", last_hsv[name][int(y * k), int(x * k)])
    if not once:
        for c in cams:
            cv2.namedWindow(c.name)
            cv2.setMouseCallback(c.name, on_mouse, c.name)
    try:
        while True:
            shots = []
            for c in cams:
                with c.lock:
                    s = c.shown
                if s is None:
                    continue
                f, res = s
                shots.append((c, f, res))
                new_result = last_recorded.get(c.name) is not res
                if new_result:
                    last_recorded[c.name] = res
                    if res["holds"] and first_detection_at is None:
                        first_detection_at = time.monotonic()
                if (capture_started is None and first_detection_at is not None
                        and time.monotonic() - first_detection_at >= capture_delay):
                    capture_started = time.monotonic()
                if new_result and capture_started is not None:
                    observations.extend((c.name, h) for h in res["holds"])
                vis, rect_vis = P.draw(f, res)
                last_hsv[c.name] = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
                small = cv2.resize(vis, (SHOW_W, int(vis.shape[0] * SHOW_W / vis.shape[1])))
                cv2.putText(small, f"{c.name} : {res['status'][:80]}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2, cv2.LINE_AA)
                if res["pose"] is not None:
                    pos = res["pose"][3]
                    cv2.putText(small, f"camera / mur : x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f} m (erreur {res['pose'][2]:.2f} px)",
                                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                if once:
                    cv2.imwrite(f"live_view_{len(shots)}.png", small)
                else:
                    cv2.imshow(c.name, small)
                    if rect_vis is not None:
                        cv2.imshow("Mur de face " + c.name, rect_vis)
            if once and len(shots) == len(cams):
                for c, f, res in shots:
                    print(f"{c.name} : {res['status']}" + (f" | position {np.round(res['pose'][3], 3)} m" if res["pose"] else ""))
                holds = detect_holds([f for _, f, _ in shots], colors, verbose=False)
                all_holds, added = add_holds_to_file(holds)
                print(f"{len(holds)} prises détectées, {len(added)} nouvelles ; {len(all_holds)} dans holds.json")
                return
            if not once and capture_started is not None and time.monotonic() - capture_started >= capture_seconds:
                detected = holds_from_observations(observations)
                all_holds, added = add_holds_to_file(detected)
                print(f"Capture de {capture_seconds:.0f} s terminée : {len(detected)} prises observées, "
                      f"{len(added)} nouvelles ; {len(all_holds)} au total dans holds.json")
                return
            k = cv2.waitKey(30) & 0xFF if not once else -1
            if k == ord("q"):
                break
            if k == ord("s") and shots:
                for i, (c, f, res) in enumerate(shots):
                    cv2.imwrite(f"raw_cam{i}.png", f)
                holds = detect_holds([f for _, f, _ in shots], colors, verbose=True)
                all_holds, added = add_holds_to_file(holds)
                print(f"{len(holds)} prises détectées, {len(added)} nouvelles ; {len(all_holds)} dans holds.json (+ raw_cam*.png)")
                poses = [{"camera": c.name, "rvec": [float(x) for x in np.ravel(res["pose"][0])],
                          "tvec": [float(x) for x in np.ravel(res["pose"][1])],
                          "position_m": [float(x) for x in res["pose"][3]], "reprojection_px": float(res["pose"][2])}
                         for c, f, res in shots if res["pose"] is not None]
                if poses:
                    json.dump(poses, open("poses.json", "w"), indent=1)
                    print(f"{len(poses)} pose(s) de caméra -> poses.json")
            if k == ord("p") and shots:
                P.print_table(shots[0][2]["holds"])
    finally:
        for c in cams:
            c.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:                      # OpenCV sans interface graphique (mode --once sur serveur)
            pass


if __name__ == "__main__":
    main()
