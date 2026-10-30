
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# 1. Charger YOLO
print("Chargement de YOLO...")
model = YOLO("/home/dani/Documents/Master2/Whackathon/runs/detect/runs/detect/prises_small_v2/weights/best.pt")
print("YOLO chargé !")

# 2. Configurer la RealSense
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640, 480,
    rs.format.bgr8,
    30
)

started = False

try:
    # 3. Démarrer la caméra
    print("Démarrage de la RealSense...")
    pipeline.start(config)
    started = True
    print("RealSense connectée !")

    # 4. Traiter les images en temps réel
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        results = model.predict(source=image,conf=0.25, device="cpu",verbose=False)
        annotated = results[0].plot()
        cv2.imshow("Whackathon - RealSense YOLO", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print("Erreur :", type(e).__name__, str(e))
    raise

finally:
    if started:
        pipeline.stop()
    cv2.destroyAllWindows()
