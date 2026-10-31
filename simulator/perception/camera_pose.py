"""Position de la caméra par rapport au mur, à partir de points du mur dont on connaît les pixels (trous numérotés,
ou coins des marqueurs ArUco)."""
import cv2
import numpy as np

from perception.geometry import BOARD_ORIGIN_X_MM, BOARD_ORIGIN_Y_MM


def approx_realsense_intrinsics(image_shape):
    """Matrice K approximative d'une RealSense D435i (champ horizontal ~69,4°), si on n'a pas la vraie."""
    height, width = image_shape[:2]
    focal = 0.7226 * width
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], np.float64)


def board_mm_to_wall_m(points_mm):
    """Points en mm dans le repère de la zone percée -> points 3D en m dans le repère du mur (z = 0)."""
    return np.column_stack([points_mm[:, 0] + BOARD_ORIGIN_X_MM, points_mm[:, 1] + BOARD_ORIGIN_Y_MM,
                            np.zeros(len(points_mm))]) / 1000.0


def camera_pose_from_points(wall_points_m, image_points_px, K):
    """Pose de la caméra à partir de points du mur (m, z = 0) et de leurs pixels : solvePnP plan (IPPE) puis affinage.
    Retourne (rvec, tvec, erreur moyenne de reprojection en px, position de la caméra dans le repère du mur en m)."""
    wall = np.asarray(wall_points_m, np.float64).reshape(-1, 3)
    pixels = np.asarray(image_points_px, np.float64).reshape(-1, 1, 2)
    K = np.asarray(K, np.float64)
    ok, rvec, tvec = cv2.solvePnP(wall, pixels, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    if len(wall) > 8:
        rvec, tvec = cv2.solvePnPRefineLM(wall, pixels, K, None, rvec, tvec)
    projected, _ = cv2.projectPoints(wall, rvec, tvec, K, None)
    error = float(np.linalg.norm(projected.reshape(-1, 2) - pixels.reshape(-1, 2), axis=1).mean())
    R, _ = cv2.Rodrigues(rvec)
    return rvec.ravel(), tvec.ravel(), error, (-R.T @ tvec).ravel()
