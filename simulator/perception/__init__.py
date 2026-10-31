"""Briques de la perception des prises, une étape par fichier.

Dans l'ordre où le pipeline s'en sert (voir perception_rgb.py et perception_aruco.py, qui les enchaînent) :

    geometry.py            dimensions du panneau, repère en mm, (col, row) d'une prise, vue de face
    colors.py              RÉGLAGES : plages de couleur des prises (c'est ici qu'on ajuste si l'éclairage change)

    board_region.py        étape 1 (sans marqueur) : où est le panneau dans l'image ?  -> masque + contour
    board_corners.py       étape 2 (sans marqueur) : ses 4 coins, recalés au sous-pixel
    aruco_markers.py       étapes 1+2 (avec marqueurs) : le panneau par ses 4 marqueurs ArUco
    hole_grid.py           étape 3 : numéroter les trous -> homographie précise pixel <-> mm
    hole_grid_fallback.py  étape 3, méthode de secours (seulement si hole_grid échoue)
    holds.py               étape 4 : les prises, par couleur, dans la vue de face
    camera_pose.py         position de la caméra par rapport au mur

    display.py             dessin des résultats et tableau des prises
"""
