import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("Whackathon - YOLO", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()