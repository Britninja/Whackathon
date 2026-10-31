"""RÉGLAGES : couleurs des prises (HSV OpenCV, H de 0 à 179).

C'est le fichier à modifier si l'éclairage ou les prises changent. Pour lire le HSV d'une prise : lance
perception_rgb.py (ou live_cameras.py) et clique dessus dans la fenêtre "Camera".

Mesures sur une photo RealSense D435i en salle très lumineuse :
  orange : H 6-9,  S 180-200, V 110-120   |   bleu : H 97-99, S 190-220, V 80-95
  vert (pâle, presque translucide) : H 38-54, S 40-80, V 80-130   |   bois du panneau : H 21-31, S 40-57
Les plages ci-dessous sont larges autour de ces mesures.

Champs de chaque couleur :
  hsv_min / hsv_max : plage HSV acceptée
  draw_bgr          : couleur utilisée pour dessiner
  offset_mm         : (dx, dy) centre de la tache -> milieu des deux boulons, mesuré sur la vraie photo (écart-type ~2 mm).
                      Absent = décalage du simulateur (SIM_HOLD_OFFSET_MM dans geometry.py)
  lab_a_max         : (vert seulement) composante a* de Lab maximale. Le vert pâle a a* <= -5, le bois a* >= -3 :
                      c'est ce critère, pas la teinte, qui sépare le vert pâle du bois.
"""
import numpy as np

# Vrai mur
COLORS = {
    "Orange": {"hsv_min": np.array([0, 120, 50]), "hsv_max": np.array([14, 255, 255]),
               "draw_bgr": (0, 140, 255), "offset_mm": (-3.1, -14.6)},
    # Le vert change énormément d'une caméra à l'autre (vif sur un téléphone, délavé sur la D435i) : plage très large.
    "Vert":   {"hsv_min": np.array([33, 25, 40]), "hsv_max": np.array([88, 255, 255]),
               "draw_bgr": (0, 255, 0), "offset_mm": (-3.0, -14.5), "lab_a_max": -5},
    "Bleu":   {"hsv_min": np.array([90, 110, 40]), "hsv_max": np.array([112, 255, 255]),
               "draw_bgr": (255, 150, 0), "offset_mm": (-3.0, -14.3)},
}

# Simulateur MuJoCo : toutes les prises sont rouge-brun sombre (H~9, S~210), le bois est à H~19, S~90.
# A passer en argument : analyze(image, colors=SIM_COLORS).
SIM_COLORS = {
    "Prise": {"hsv_min": np.array([3, 130, 25]), "hsv_max": np.array([15, 255, 255]), "draw_bgr": (0, 140, 255)},
}

# Ecart minimal de a* (Lab) entre une prise verte et le bois autour d'elle. Sur une image délavée, le vert n'est
# qu'à ~8 du bois, c'est-à-dire dans le bruit : on préfère rater une prise que d'en inventer une. 0 = désactivé.
GREEN_MIN_CONTRAST = 15
