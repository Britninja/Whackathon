"""Prend UNE image couleur (1920x1080) de chaque RealSense branchée et l'enregistre : raw_cam0.png, raw_cam1.png.
Pas de fenêtre, pas de touche : aucun problème de focus sous WSL.

    uv run --with opencv-python --with pyrealsense2 python grab_raw.py
(Si tu as dû utiliser sudo pour la caméra :  sudo $(which uv) run --with opencv-python --with pyrealsense2 python grab_raw.py)
"""
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

serials = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()][:2]
if not serials:
    sys.exit("Aucune RealSense trouvée (voir les étapes usbipd / lsusb).")
for i, s in enumerate(serials):
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(s)
    cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)      # couleur seule
    pipe.start(cfg)
    try:
        for _ in range(45):                                   # on laisse l'exposition automatique se stabiliser (~1,5 s)
            frames = pipe.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data()).copy()
    finally:
        pipe.stop()
    cv2.imwrite(f"raw_cam{i}.png", img)
    print(f"caméra {s} -> raw_cam{i}.png ({img.shape[1]}x{img.shape[0]})")