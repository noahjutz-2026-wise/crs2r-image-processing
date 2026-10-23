"""Synthetic fisheye-scene generator used to test the state estimation library.

It renders frames of the cyberrunner labyrinth as they would appear through
the calibrated fisheye camera, by projecting the known 3D geometry (fixed
outer dots, maze corner dots, ball) through the OcamModel, so the pipeline can
be validated against exact ground truth.
"""

import cv2
import numpy as np

from crs2r_image_processing.state_estimation.plate_pose import PlatePoseEstimator

# camera looks straight down onto the maze:
# camera frame axes expressed in world frame: x_C=(1,0,0), y_C=(0,-1,0), z_C=(0,0,-1)
R_WC_DOWN = np.column_stack(
    [(1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)]
)
CAM_HEIGHT = 0.45

# ocam ray <-> camera-frame direction converter (as used by the pipeline:
# see Measurements.create_mask: camera_points = cam2world(pix)[:, [1, 0, 2]], z *= -1)
S = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)

BG = (120, 120, 120)  # gray: hue/sat out of both detection ranges
DOT = (0, 255, 0)  # green: corner + fixed dot color
BALL = (255, 255, 0)  # cyan: ball color
DOT_RADIUS = 4
BALL_RADIUS = 7


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def R__W_M(alpha, beta):
    """Maze orientation in world for Euler angles (alpha around +X, beta around +Y).

    Consistent with PlatePoseEstimator.getXYAnglesFrom_R__W_M.
    """
    return rot_y(beta) @ rot_x(alpha)


def camera_t():
    return np.array(
        [PlatePoseEstimator.L_EXT_INT_X / 2, PlatePoseEstimator.L_EXT_INT_Y / 2, CAM_HEIGHT]
    )


def project(plate_pose, P_w, R_wc=R_WC_DOWN, t_w=None):
    """Project world points (N,3) to raw fisheye pixel coords (row, col).

    (world2cam output convention: first coord is the row/line, second the column)
    """
    if t_w is None:
        t_w = camera_t()
    P_w = np.atleast_2d(np.asarray(P_w, float))
    d_c = (R_wc.T @ (P_w - t_w).T).T
    w = d_c @ S.T  # w_i = S @ d_c_i
    return plate_pose.o.world2cam(w)


def scene_geometry(alpha=0.0, beta=0.0, ball_m=np.array([0.08, 0.11]), extra_dots=()):
    """Return world coordinates of all rendered dots for a given plate tilt.

    The maze frame origin sits at the center of the outer-dot rectangle in the
    world (as in the real rig), so the maze corner dots stay inside the masked
    region. extra_dots: iterable of maze-frame positions (3,) rendered as
    green dots.
    """
    t0 = np.array(
        [PlatePoseEstimator.L_EXT_INT_X / 2, PlatePoseEstimator.L_EXT_INT_Y / 2, 0.0]
    )
    T__W_M = np.eye(4)
    T__W_M[:3, :3] = R__W_M(alpha, beta)
    T__W_M[:3, 3] = t0
    fix_w = PlatePoseEstimator.MODEL_POINTS_FIXED_CORNERS.astype(float)  # world frame
    cor_m = np.hstack(
        (PlatePoseEstimator.MODEL_POINTS_CORNERS.astype(float), np.ones((4, 1)))
    )
    cor_w = (T__W_M @ cor_m.T).T[:, :3]
    ball_w = (T__W_M @ np.array([ball_m[0], ball_m[1], PlatePoseEstimator.R_BALL, 1.0]))[:3]
    extra_w = [(T__W_M @ np.array([p[0], p[1], p[2], 1.0]))[:3] for p in extra_dots]
    return fix_w, cor_w, ball_w, extra_w, T__W_M


def render_frame(plate_pose, alpha=0.0, beta=0.0, ball_m=np.array([0.15, 0.11]), extra_dots=()):
    """Render a (400, 640, 3) uint8 BGR fisheye frame of the scene."""
    fix_w, cor_w, ball_w, extra_w, _ = scene_geometry(alpha, beta, ball_m, extra_dots)
    frame = np.full((400, 640, 3), BG, dtype=np.uint8)
    for p in np.vstack((fix_w, cor_w, np.array(extra_w) if extra_w else np.zeros((0, 3)))):
        v, u = project(plate_pose, p)[0]  # (row, col)
        cv2.circle(frame, (int(round(u)), int(round(v))), DOT_RADIUS, DOT, -1)
    v, u = project(plate_pose, ball_w)[0]
    cv2.circle(frame, (int(round(u)), int(round(v))), BALL_RADIUS, BALL, -1)
    return frame


def markers_csv(plate_pose):
    """Marker coordinates in the format of the original markers.csv:
    4 outer fixed dots then 4 maze corner dots, rows in (u, v) = (col, row).
    Generated with the plate flat, as the select_markers tool requires.
    """
    fix_w, cor_w, _, _, _ = scene_geometry(0.0, 0.0)
    px = np.vstack((project(plate_pose, fix_w), project(plate_pose, cor_w)))
    return px[:, ::-1]  # (col, row)
