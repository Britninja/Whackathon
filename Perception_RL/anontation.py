
import cv2
from pathlib import Path



BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset-2"

NOM_IMAGE = "image_028.jpg"
CLASSE = 0  # prise d'escalade



chemin_image = None
chemin_label = None

for separation in ["train", "val"]:

    candidat = DATASET / "images" / separation / NOM_IMAGE

    if candidat.exists():
        chemin_image = candidat
        chemin_label = (
            DATASET / "labels" / separation / "image_028.txt"
        )
        break

if chemin_image is None:
    raise FileNotFoundError(
        f"Impossible de trouver {NOM_IMAGE} dans train ou val"
    )

# Créer le dossier labels si nécessaire
chemin_label.parent.mkdir(parents=True, exist_ok=True)



image = cv2.imread(str(chemin_image))

if image is None:
    raise RuntimeError("Impossible de lire l'image")

hauteur, largeur = image.shape[:2]

annotations = []
affichage = image.copy()

print(f"Image : {chemin_image}")
print("Dessine un rectangle autour de chaque prise.")
print("ENTREE ou ESPACE : valider un rectangle")
print("C : annuler le rectangle en cours")
print("ECHAP : terminer l'annotation")

# ================= ANNOTATION =================

while True:

    x, y, w, h = cv2.selectROI(
        "Annotation des prises",
        affichage,
        showCrosshair=True,
        fromCenter=False
    )

    if w == 0 or h == 0:
        break

    x_centre = (x + w / 2) / largeur
    y_centre = (y + h / 2) / hauteur
    largeur_box = w / largeur
    hauteur_box = h / hauteur
    annotations.append(
        f"{CLASSE} "
        f"{x_centre:.6f} "
        f"{y_centre:.6f} "
        f"{largeur_box:.6f} "
        f"{hauteur_box:.6f}"
    )

    cv2.rectangle(
        affichage,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    print(f"Prise {len(annotations)} annotée")

cv2.destroyAllWindows()


if annotations:

    with open(chemin_label, "w") as fichier:
        fichier.write("\n".join(annotations) + "\n")

    print("\nAnnotation terminée !")
    print(f"Nombre de prises : {len(annotations)}")
    print(f"Fichier généré : {chemin_label}")

else:
    print("Aucune prise annotée : rien n'a été enregistré.")
