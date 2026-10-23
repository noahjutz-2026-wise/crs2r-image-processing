from crs2r_image_processing.state_estimation.pipeline import EstimationPipeline
from crs2r_image_processing.state_estimation.measurements import Measurements
from crs2r_image_processing.state_estimation.plate_pose import PlatePoseEstimator
from crs2r_image_processing.state_estimation.ocam import OcamModel
from crs2r_image_processing.state_estimation.detection import Detector, DetectorFixedPts
from crs2r_image_processing.state_estimation.estimator import KF, KFBias, FiniteDiff


class TiltAngleEstimator:
    """Estimates plate tilt angles (alpha, beta) from camera frames."""

    def __init__(self, markers=None):
        if markers is None:
            from tests.synth import markers_csv
            plate_pose = PlatePoseEstimator()
            markers = markers_csv(plate_pose)
        self.pipeline = EstimationPipeline(fps=60.0, markers=markers, estimator="FiniteDiff")

    def estimate_tilt(self, frame):
        """Given a camera frame, preprocess and return (alpha, beta) in radians.

        Accepts either raw 1280x720 (as captured from See3CAM) or 400x640 frames.
        """
        import cv2

        if frame.shape[:2] != (400, 640):
            resized = cv2.resize(frame, (640, 360))
            frame = cv2.copyMakeBorder(resized, 20, 20, 0, 0, cv2.BORDER_CONSTANT, 0)
        _, _, (alpha, beta), _, _ = self.pipeline.estimate(frame)
        return float(alpha), float(beta)


__all__ = [
    "TiltAngleEstimator",
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
