import numpy as np                 # numpy : calcul avec des vecteurs et des matrices
import placo                       # la bibliothèque de cinématique de Rhoban
import microban_model              # le modèle du robot (le même que dans le simulateur)

# Script pour tester l'import et l'utilisation de placo avec le modele du Robot -> ok

# ------------------------------------------------------------------
# 1. CHARGER LE ROBOT
# ------------------------------------------------------------------
# microban_model.urdf(climb=True) donne le chemin du fichier qui décrit le robot
# (ses segments, ses articulations, leurs axes, leurs limites).
# climb=True : version "grimpeur" (mains-boules, pieds d'escalade).
# str(...) : Placo veut du texte, pas un objet "chemin" Python.
robot = placo.RobotWrapper(str(microban_model.urdf(climb=True)))

# ------------------------------------------------------------------
# 2. CRÉER LE SOLVEUR D'IK
# ------------------------------------------------------------------
# Le solveur est le "cerveau" : on lui donne des objectifs (tâches),
# il calcule les angles des articulations pour les atteindre au mieux.
solver = placo.KinematicsSolver(robot)

# Par défaut, Placo considère que le tronc flotte librement dans l'espace
# (il peut se déplacer et tourner). Ici on le FIGE : seuls les bras et les
# jambes bougent. Plus simple pour un premier test.
solver.mask_fbase(True)

# Calcule la position de toutes les parties du robot à partir des angles actuels.
# À appeler avant de lire des positions.
robot.update_kinematics()


# ------------------------------------------------------------------
# 3. UNE PETITE FONCTION POUR LIRE LA POSITION DE LA MAIN
# ------------------------------------------------------------------
def hand():
    # get_T_world_frame donne une matrice 4x4 (position + orientation)
    # de la frame "right_hand_ball", c'est-à-dire le centre de la boule
    # de la main droite. [:3, 3] garde seulement la position (x, y, z),
    # en mètres. .copy() évite de modifier la valeur d'origine par erreur.
    return robot.get_T_world_frame("right_hand_ball")[:3, 3].copy()


# Tous les angles valent 0 au départ : la main est à ce point (x, y, z).
print("main droite, tout à 0 :", hand())

# ------------------------------------------------------------------
# 4. DONNER UN OBJECTIF AU SOLVEUR
# ------------------------------------------------------------------
# On crée une tâche de POSITION : "la frame right_hand_ball doit être au
# point demandé". Au début, on lui donne la position actuelle de la main.
task = solver.add_position_task("right_hand_ball", hand())

# "soft" = objectif souple : si le solveur ne peut pas tout satisfaire,
# il fait un compromis. 1.0 = poids (importance) de cette tâche.
task.configure("main", "soft", 1.0)

# Nouvelle cible = position actuelle + 3 cm vers le haut (z, en mètres).
target = hand() + np.array([0.0, 0.0, 0.03])
task.target_world = target          # on donne la cible à la tâche

# ------------------------------------------------------------------
# 5. RÉSOUDRE (LANCER L'IK)
# ------------------------------------------------------------------
# Chaque solve() fait UN petit pas vers la solution (il corrige un peu
# les angles). On le répète 100 fois pour converger.
for _ in range(100):
    solver.solve(True)              # True = on applique le pas aux angles du modèle
    robot.update_kinematics()       # on recalcule où sont les parties du robot

# ------------------------------------------------------------------
# 6. LIRE LE RÉSULTAT
# ------------------------------------------------------------------
print("cible   :", target)          # où on voulait la main
print("atteint :", hand())          # où elle est vraiment après l'IK
# Les angles trouvés par Placo pour le bras droit (en radians) :
for name in ("right_shoulder_pitch", "right_shoulder_roll", "right_elbow"):
    print(name, robot.get_joint(name))