import cv2
import numpy as np

from crs2r_image_processing.state_estimation.pipeline import EstimationPipeline
from crs2r_image_processing.state_estimation.measurements import Measurements
from crs2r_image_processing.state_estimation.plate_pose import PlatePoseEstimator
from crs2r_image_processing.state_estimation.ocam import OcamModel
from crs2r_image_processing.state_estimation.detection import Detector, DetectorFixedPts
from crs2r_image_processing.state_estimation.estimator import KF, KFBias, FiniteDiff


def generate_default_markers():
    """Generate default marker pixel positions using known CAD geometry and camera height."""
    plate_pose = PlatePoseEstimator()
    t0 = np.array([PlatePoseEstimator.L_EXT_INT_X / 2, PlatePoseEstimator.L_EXT_INT_Y / 2, 0.0])
    cam_height = 0.45
    t_cam = np.array([PlatePoseEstimator.L_EXT_INT_X / 2, PlatePoseEstimator.L_EXT_INT_Y / 2, cam_height])
    r_wc = np.column_stack([(1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)])
    s = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)

    fix_w = PlatePoseEstimator.MODEL_POINTS_FIXED_CORNERS.astype(float)
    cor_w = PlatePoseEstimator.MODEL_POINTS_CORNERS.astype(float) + t0
    pts_w = np.vstack((fix_w, cor_w))

    d_c = (r_wc.T @ (pts_w - t_cam).T).T
    w = d_c @ s.T
    pix = plate_pose.o.world2cam(w)  # (row, col)
    return pix[:, ::-1]  # (col, row)


class TiltAngleEstimator:
    """Estimates plate tilt angles (alpha, beta) from camera frames."""

    def __init__(self, markers=None):
        if markers is None:
            markers = generate_default_markers()
        self.pipeline = EstimationPipeline(fps=55.0, markers=markers, estimator="KF")

    def estimate_tilt(self, frame):
        """Given a camera frame, preprocess and return (alpha, beta) in radians.

        Accepts either raw 1280x720 / 1920x1080 frames or already formatted 400x640 frames.
        """
        if frame.shape[:2] != (400, 640):
            resized = cv2.resize(frame, (640, 360))
            frame = cv2.copyMakeBorder(resized, 20, 20, 0, 0, cv2.BORDER_CONSTANT, 0)
        _, _, (alpha, beta), _, _ = self.pipeline.estimate(frame)
        return float(alpha), float(beta)


__all__ = [
    "TiltAngleEstimator",
    "generate_default_markers",
    "EstimationPipeline",
    "Measurements",
    "PlatePoseEstimator",
    "OcamModel",
    "Detector",
    "DetectorFixedPts",
    "KF",
    "KFBias",
    "FiniteDiff",
]
