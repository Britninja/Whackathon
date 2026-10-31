"""Affichage des résultats de analyze() : image annotée, vue de face annotée, tableau des prises."""
import cv2
import numpy as np


def draw(image, result):
    """Retourne (image caméra annotée, vue de face annotée ou None)."""
    vis = image.copy()
    if result["mask"] is not None:
        contours, _ = cv2.findContours(result["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 0), 1)
    if result["quad"] is not None:
        cv2.polylines(vis, [result["quad"].astype(int)], True, (0, 255, 255), 3)
        for (x, y), inside, label in zip(result["quad"], result["inside"] or [True] * 4, ["HG", "HD", "BD", "BG"]):
            cv2.circle(vis, (int(x), int(y)), 9, (0, 255, 0) if inside else (0, 0, 255), -1)
            cv2.putText(vis, label, (int(x) + 12, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    front_vis = None
    if result["rect"] is not None:
        front_vis = result["rect"].copy()
        front_to_px = np.linalg.inv(result["H_px2rect"])
        for hold in result["holds"]:
            cv2.circle(front_vis, (int(hold["u"]), int(hold["v"])), 7, hold["bgr"], 2)
            cv2.putText(front_vis, f'({hold["col"]},{hold["row"]})', (int(hold["u"]) + 9, int(hold["v"]) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            p = cv2.perspectiveTransform(np.float32([[[hold["u"], hold["v"]]]]), front_to_px)[0, 0]
            cv2.circle(vis, (int(p[0]), int(p[1])), 8, hold["bgr"], 2)
    ok = result["status"].startswith("OK")
    cv2.putText(vis, result["status"], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if ok else (0, 0, 255), 2)
    return vis, front_vis


def print_table(holds):
    print(f"--- {len(holds)} prises ---")
    for h in sorted(holds, key=lambda h: (h["row"], h["col"])):
        print(f'{h["name"]:7s} col={h["col"]:3d} row={h["row"]:3d}  x={h["x"]:6.0f} mm  y={h["y"]:6.0f} mm')
