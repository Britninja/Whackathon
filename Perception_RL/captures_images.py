
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path

# ================= CONFIGURATION =================

LARGEUR = 640
HAUTEUR = 480
FPS = 30

# Dossier de sauvegarde
dossier = Path(__file__).resolve().parent / "images"
dossier.mkdir(parents=True, exist_ok=True)

# ================= NUMEROTATION =================

numeros = []

for fichier in dossier.glob("image_*.jpg"):
    numero = fichier.stem.replace("image_", "")

    if numero.isdigit():
        numeros.append(int(numero))

compteur = max(numeros, default=0)
captures = 0

# ================= REALSENSE =================

pipeline = rs.pipeline()
config = rs.config()

# Activer uniquement la caméra couleur
config.enable_stream(
    rs.stream.color,
    LARGEUR,
    HAUTEUR,
    rs.format.bgr8,
    FPS
)

started = False

try:
    # Démarrer la RealSense
    pipeline.start(config)
    started = True

    print("Camera RealSense connectee !")
    print("ENTREE : capturer une photo")
    print("Q : quitter")
    print("CTRL+C : arreter depuis le terminal")
    print("Aucune limite de photos")

    # ================= BOUCLE PRINCIPALE =================

    while True:

        # Recuperer une image de la RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convertir en tableau NumPy (compatible OpenCV)
        frame = np.asanyarray(color_frame.get_data())

        # Copie pour afficher les instructions
        affichage = frame.copy()

        cv2.putText(
            affichage,
            f"Photos capturees : {captures}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            affichage,
            "ENTREE = Photo | Q = Quitter",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Afficher la vidéo en direct
        cv2.imshow(
            "Whackathon - Capture RealSense",
            affichage
        )

        touche = cv2.waitKey(1) & 0xFF

        # ================= CAPTURE =================

        if touche in (10, 13):

            prochain = compteur + 1

            nom = dossier / f"image_{prochain:03d}.jpg"

            # Sauvegarde de l'image originale
            if cv2.imwrite(str(nom), frame):

                compteur = prochain
                captures += 1

                print(
                    f"Photo {captures} enregistree : {nom.name}"
                )

            else:
                print("Erreur pendant l'enregistrement")

        # ================= QUITTER =================

        elif touche == ord("q"):
            print("Arret demande")
            break

except KeyboardInterrupt:
    print("\nArret avec Ctrl+C")

finally:
    if started:
        pipeline.stop()

    cv2.destroyAllWindows()

print(f"\n{captures} nouvelles photos enregistrees.")
print(f"Dossier : {dossier}")
