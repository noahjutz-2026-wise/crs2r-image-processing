import cv2
import numpy as np

from crs2r_image_processing.state_estimation.detection import Detector, DetectorFixedPts
from crs2r_image_processing.state_estimation.plate_pose import PlatePoseEstimator


class Measurements:
    def __init__(self, markers, show_subimages_detector=False):
        self.detector = Detector(markers[4:], show_subimages=show_subimages_detector)
        self.detector_fixed_points = DetectorFixedPts(
            markers[:4], show_subimages=show_subimages_detector
        )
        self.plate_pose = PlatePoseEstimator()

        self.plate_angles = (None, None)
        self.ball_pos = None
        self.ball_img_coords = None
        self.ball_subimg = None

        self.SUBIMG_SIZE = 64
        # half-size of the ball subimage in the maze frame, in mm
        # (the original implementation samples 1e-3 * mgrid[-32:32])
        self.SUBIMG_HALF_RANGE_MM = 32

    def process_frame(
        self,
        frame,
        get_ball_subimg=False,
    ):
        """
        Process the frame to compute the angles of the plate and the position of the ball in the maze frame {m}.

        Args :
            frame: np.ndarray, dim: (400, 640)
        """
        if self.plate_pose.T__W_C is None:
            self.camera_localization(frame)

        frame = cv2.bitwise_and(frame, frame, mask=self.mask)
        if get_ball_subimg:
            frame_copy = frame.copy()

        corners_img_coords, ball_img_coords = self.detector.process_frame(
            frame
        )  # (x,y)

        raw_pts = np.zeros((5, 2))
        raw_pts[:4, :] = corners_img_coords
        raw_pts[4, :] = ball_img_coords
        undist_pts = self.plate_pose.undistort_points(raw_pts)  # (x,y)
        corners_undist = undist_pts[:4, :]
        ball_undist = undist_pts[4, :]

        alpha, beta = self.plate_pose.estimate_anglesXY(corners_undist)
        self.plate_angles = (alpha, beta)
        self.ball_pos = self.ball_pos_backproject(
            ball_undist, self.plate_pose.K, self.plate_pose.T__C_M
        )
        self.ball_img_coords = ball_img_coords
        if get_ball_subimg:
            self.ball_subimg = self.extract_ball_subimg(frame_copy, self.ball_pos)

    def extract_ball_subimg(self, frame, ball_pos):
        """
        Extract a 64x64 subimage centered on the ball. The subimage is sampled
        on the (tilted) maze plane in the maze frame {m}, over a square of
        +/- SUBIMG_HALF_RANGE_MM (i.e. +/- 3.2 cm) around the ball position,
        using the original fisheye camera model for the mapping.

        Args :
            frame: np.ndarray, dim: (400, 640)
                   the (masked) frame to sample from.
            ball_pos: np.ndarray, dim: (3,)
                      3d position of the ball in the maze frame {m}.

        Returns :
            subimg: np.ndarray, dim: (64, 64, 3)
                    if the ball position is NaN, a black (64, 64, 3) image.
        """
        if np.isnan(ball_pos[0]):
            return np.zeros((self.SUBIMG_SIZE, self.SUBIMG_SIZE, 3), dtype=np.uint8)
        size = self.SUBIMG_SIZE
        points_board = np.zeros((size * size, 4))
        points_board[:, -1] = 1.0
        half = self.SUBIMG_HALF_RANGE_MM
        points_board[:, :2] = (
            1.0e-3 * np.mgrid[-half:half, -half:half][::-1].reshape(2, -1).transpose()
        )
        points_board[:, 1] *= -1
        points_board[:, :3] += ball_pos
        points_cam = (self.plate_pose.T__C_M @ points_board.T).T[:, :3]
        points_cam[:, :2] = points_cam[:, [1, 0]]
        points_cam[:, 2] *= -1
        points_cam = self.plate_pose.o.world2cam(points_cam)
        points_cam = points_cam.reshape(size, size, 2).astype(np.float32)
        subimg = cv2.remap(
            frame, points_cam[..., 1], points_cam[..., 0], cv2.INTER_LINEAR
        )
        return subimg

    def get_ball_subimg(self):
        return self.ball_subimg

    def get_ball_coordinates(self):
        """
        Return the pixel coordinates of the ball in the image frame {m}.

        Returns:
            ball_pos: np.ndarray, dim: (2,)
                    2d position of the ball in the image frame.

        """
        return self.ball_img_coords

    def get_ball_position_in_maze(self):
        """
        Return the position of the ball in the maze frame {m}.

        Returns:
            ball_pos: np.ndarray, dim: (3,)
                    3d position of the ball in the maze frame {m}.
                    note: the z-coordinate of the ball in maze frame is fixed and known
                    by assumption of constant contact with the maze: z__m_b = ball_radius.
                    The maze frame {m} is attached to the (possibly tilted) plate, so the
                    (x, y) coordinates are already expressed in the plate's own frame,
                    i.e. they are normalized with respect to the plate angle.

        """
        return self.ball_pos

    def get_plate_pose(self):
        """
        Return the angles (Euler YXZ) that describe the orientation of the maze frame {m} wrt the world frame {w}.

        Returns:
            ball_pos: Tuple(float, float)
                      (alpha, beta) around X and Y respectively
        """
        return self.plate_angles

    def camera_localization(self, frame):
        """
        Compute the pose of the camera {c} wrt to the world frame {w} : T__W_C.
        """
        fix_pts = self.detector_fixed_points.detect_corners(frame)
        self.detector.fixed_corners = fix_pts
        self.plate_pose.camera_localization(fix_pts)
        self.create_mask(frame)

    def create_mask(self, frame):
        h, w = frame.shape[:2]
        coords = np.mgrid[0:h, 0:w].transpose(1, 2, 0).reshape(-1, 2)
        camera_points = self.plate_pose.o.cam2world(coords)[:, [1, 0, 2]]
        camera_points[:, 2] *= -1
        world_vec = (self.plate_pose.T__W_C[:3, :3] @ camera_points.T).T
        world_vec = world_vec / world_vec[:, 2:]
        world_points = (
            world_vec * (-self.plate_pose.T__W_C[2, -1])
            + self.plate_pose.T__W_C[:3, -1]
        )
        mask = (
            (world_points[:, 0] >= -(2.0 * self.plate_pose.r))
            & (
                world_points[:, 0]
                <= self.plate_pose.L_EXT_INT_X + 2.0 * self.plate_pose.r
            )
            & (world_points[:, 1] >= -(2.0 * self.plate_pose.r))
            & (
                world_points[:, 1]
                <= self.plate_pose.L_EXT_INT_Y + 2.0 * self.plate_pose.r
            )
        )
        self.mask = 255 * mask.reshape(h, w, 1).astype(np.uint8)

    def ball_pos_backproject(self, ball_undist, K, T__C_M):
        """
        Compute the 3d position of the ball in the maze frame {m}.
        Returns:
            x_M: np.ndarray, dim: (3,)
                 3d position of the ball in the maze frame {m}.
                 note: the z-coordinate of the ball in maze frame is fixed and known
                 by assumption of constant contact with the maze: z__m_b = ball_radius.
        """
        if np.any(ball_undist == np.nan):
            return np.array([np.nan, np.nan, np.nan])
        d = PlatePoseEstimator.R_BALL
        v, u = ball_undist
        H = K @ T__C_M[:3, :]
        h_11, h_12, h_13, h_14 = H[0, :]
        h_21, h_22, h_23, h_24 = H[1, :]
        h_31, h_32, h_33, h_34 = H[2, :]
        A = np.array(
            [[u * h_31 - h_11, u * h_32 - h_12], [v * h_31 - h_21, v * h_32 - h_22]]
        )
        b = np.array(
            [
                d * h_13 + h_14 - d * u * h_33 - u * h_34,
                d * h_23 + h_24 - d * v * h_33 - v * h_34,
            ]
        )
        x = np.linalg.solve(A, b)
        x_M = np.array([x[0], x[1], PlatePoseEstimator.R_BALL])
        return x_M
