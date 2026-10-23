import cv2
import numpy as np
import pytest
from importlib.resources import files

import synth
from crs2r_image_processing.state_estimation import (
    Detector,
    DetectorFixedPts,
    EstimationPipeline,
    FiniteDiff,
    KF,
    KFBias,
    Measurements,
    OcamModel,
    PlatePoseEstimator,
)

@pytest.fixture(scope="module")
def plate_pose():
    return PlatePoseEstimator()


@pytest.fixture(scope="module")
def markers(plate_pose):
    return synth.markers_csv(plate_pose)


@pytest.fixture(scope="module")
def pipeline(plate_pose, markers):
    return EstimationPipeline(fps=55.0, markers=markers, estimator="KF")


def test_ocam_roundtrip(plate_pose):
    """world2cam inverts cam2world on real image pixels."""
    rng = np.random.default_rng(0)
    pts = np.column_stack(
        (rng.uniform(0, 400, 200), rng.uniform(0, 640, 200))
    )
    w = plate_pose.o.cam2world(pts)
    err = np.abs(plate_pose.o.world2cam(w) - pts).max()
    assert err < 0.05


def test_ocam_scale():
    o = OcamModel(_calib_path())
    xc0, yc0, h0, w0 = o.xc, o.yc, o.height, o.width
    ss0 = o.ss.copy()
    o.scale(3)
    assert np.isclose(o.xc, xc0 / 3)
    assert np.isclose(o.yc, yc0 / 3)
    assert (o.height, o.width) == (400, 640)
    # polynomial coefficients scale as ss[i] *= factor**(i-1)
    expected = ss0 * np.array([3.0 ** (i - 1) for i in range(5)])
    assert np.allclose(o.ss, expected)


def _calib_path():
    return str(
        files("crs2r_image_processing").joinpath(
            "state_estimation/data/calib_results_cyberrunner.txt"
        )
    )


def test_unfisheye(plate_pose):
    """o.undistort() maps a dot from the fisheye frame to its pinhole position."""
    ball_m = np.array([0.08, 0.11])
    frame = synth.render_frame(plate_pose, 0.0, 0.0, ball_m=ball_m)
    _, _, ball_w, _, _ = synth.scene_geometry(0.0, 0.0, ball_m=ball_m)
    src_px = synth.project(plate_pose, ball_w)[0]

    und = plate_pose.o.undistort(frame)
    assert und.shape == (400, 640, 3)

    # replicate the grid mapping used inside OcamModel.undistort to predict
    # where the dot must appear in the undistorted image
    o = plate_pose.o
    h, w = int(o.height), int(o.width)
    xs = np.linspace(
        min(o.edges[0, 0], o.edges[1, 0]), max(o.edges[2, 0], o.edges[3, 0]), num=h
    )
    ys = np.linspace(
        min(o.edges[0, 1], o.edges[2, 1]), max(o.edges[1, 1], o.edges[3, 1]), num=w
    )
    xg, yg = np.meshgrid(xs, ys, indexing="ij")
    grid = np.stack([xg.ravel(), yg.ravel(), -np.ones(h * w)], axis=1)
    pc = o.world2cam(grid)
    k = int(np.argmin(np.linalg.norm(pc - src_px, axis=1)))
    i, j = k // w, k % w
    assert np.linalg.norm(pc[k] - src_px) < 2.0

    hsv = cv2.cvtColor(und, cv2.COLOR_BGR2HSV)
    # the cyan ball dot must be centered on the predicted undistorted pixel
    win = hsv[i - 8 : i + 9, j - 8 : j + 9]
    cy, cx = np.where((win[..., 0] >= 85) & (win[..., 0] <= 125) & (win[..., 1] > 125))
    assert len(cy) > 20
    assert abs(cy.mean() + (i - 8) - i) < 3
    assert abs(cx.mean() + (j - 8) - j) < 3


def test_pipeline_flat(pipeline, plate_pose):
    frame = synth.render_frame(plate_pose, 0.0, 0.0, ball_m=np.array([0.08, 0.11]))
    x_hat, P, inputs, xb, yb = pipeline.estimate(frame)
    alpha, beta = np.degrees(inputs)
    assert abs(alpha) < 0.2 and abs(beta) < 0.2
    assert abs(xb - 0.08) < 0.002 and abs(yb - 0.11) < 0.002
    assert x_hat.shape == (4,) and P.shape == (4, 4)


def test_pipeline_tilted(pipeline, plate_pose):
    """Untwist: plate angles are recovered under tilt and the ball position is
    expressed in the (rotated) maze frame."""
    alpha_t, beta_t = 4.0, -3.0
    ball_m = np.array([0.10, 0.10])
    frame = synth.render_frame(
        plate_pose, np.radians(alpha_t), np.radians(beta_t), ball_m=ball_m
    )
    x_hat, P, inputs, subimg, xb, yb = pipeline.estimate(frame, return_ball_subimg=True)
    alpha, beta = np.degrees(inputs)
    assert abs(alpha - alpha_t) < 0.25 and abs(beta - beta_t) < 0.25
    assert abs(xb - ball_m[0]) < 0.002 and abs(yb - ball_m[1]) < 0.002


def test_ball_subimg(pipeline, plate_pose):
    """64x64 subimage centered on the ball, sampled on the tilted plate plane
    over +/- 3.2 cm (the original 1e-3 * mgrid[-32:32] range)."""
    alpha_t, beta_t = 4.0, -3.0
    ball_m = np.array([0.10, 0.10])
    # marker dot placed at maze offset (+1.5 cm, -1.5 cm) from the ball
    frame = synth.render_frame(
        plate_pose,
        np.radians(alpha_t),
        np.radians(beta_t),
        ball_m=ball_m,
        extra_dots=[
            np.array(
                [ball_m[0] + 0.015, ball_m[1] - 0.015, PlatePoseEstimator.R_BALL]
            )
        ],
    )
    x_hat, P, inputs, subimg, xb, yb = pipeline.estimate(
        frame, return_ball_subimg=True
    )
    assert subimg.shape == (64, 64, 3) and subimg.dtype == np.uint8
    hsv = cv2.cvtColor(subimg, cv2.COLOR_BGR2HSV)
    # ball (cyan) centered
    cy, cx = np.where((hsv[..., 0] >= 85) & (hsv[..., 0] <= 125) & (hsv[..., 1] > 125))
    assert abs(cy.mean() - 32) < 1.5 and abs(cx.mean() - 32) < 1.5
    # marker dot placed at maze offset (+1.5 cm, -1.5 cm) appears at
    # subimg (row, col) = (32 + 15, 32 + 15): the subimage is untwisted
    gy, gx = np.where((hsv[..., 0] > 40) & (hsv[..., 0] < 80) & (hsv[..., 1] > 125))
    assert abs(gy.mean() - 47) < 2 and abs(gx.mean() - 47) < 2


def test_ball_not_found(pipeline, plate_pose):
    """No ball in the frame -> NaN measurement, black subimg, estimator runs."""
    # render without ball: frame with only the dots
    fix_w, cor_w, _, _, _ = synth.scene_geometry(0.0, 0.0)
    f = np.full((400, 640, 3), synth.BG, dtype=np.uint8)
    for p in np.vstack((fix_w, cor_w)):
        v, u = synth.project(plate_pose, p)[0]
        cv2.circle(f, (int(round(u)), int(round(v))), synth.DOT_RADIUS, synth.DOT, -1)
    x_hat, P, inputs, subimg, xb, yb = pipeline.estimate(f, return_ball_subimg=True)
    x_hat2, P2, inputs2, subimg2, xb2, yb2 = pipeline.estimate(
        f, return_ball_subimg=True
    )
    assert np.isnan(xb2) and np.isnan(yb2)
    assert subimg.shape == (64, 64, 3) and subimg.sum() == 0
    # KF prediction step still ran (state update without measurement)
    assert x_hat2.shape == (4,) and np.all(np.isfinite(x_hat2))


def test_ball_tracking(pipeline, plate_pose):
    """Predictive cropping keeps detecting the ball as it moves."""
    positions = [(0.08, 0.11), (0.09, 0.11), (0.10, 0.11), (0.11, 0.11)]
    for x, y in positions:
        frame = synth.render_frame(plate_pose, 0.0, 0.0, ball_m=np.array([x, y]))
        x_hat, P, inputs, xb, yb = pipeline.estimate(frame)
        assert abs(xb - x) < 0.002 and abs(yb - y) < 0.002


def test_mask(pipeline, plate_pose):
    """Pixels outside the maze region are masked out of the frame."""
    frame = synth.render_frame(plate_pose, 0.0, 0.0, ball_m=np.array([0.08, 0.11]))
    pipeline.estimate(frame)
    mask = pipeline.measurements.mask
    assert mask.shape == (400, 640, 1) and mask.dtype == np.uint8
    # ball pixel inside -> kept; far corner pixel -> masked
    assert mask[225, 250, 0] == 255
    assert mask[100, 100, 0] == 0
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    assert masked[100, 100].sum() == 0


def test_measurements_lower_level(plate_pose, markers):
    """Measurements exposes the lower-level getters directly."""
    m = Measurements(markers)
    frame = synth.render_frame(
        plate_pose, np.radians(4), np.radians(-3), ball_m=np.array([0.10, 0.10])
    )
    m.process_frame(frame, get_ball_subimg=True)
    alpha, beta = m.get_plate_pose()
    assert abs(np.degrees(alpha) - 4) < 0.25 and abs(np.degrees(beta) + 3) < 0.25
    pos = m.get_ball_position_in_maze()
    assert abs(pos[0] - 0.10) < 0.002 and abs(pos[1] - 0.10) < 0.002
    assert np.isclose(pos[2], PlatePoseEstimator.R_BALL)
    img_coords = m.get_ball_coordinates()
    assert img_coords.shape == (2,)
    subimg = m.get_ball_subimg()
    assert subimg.shape == (64, 64, 3)


def test_plate_pose_estimator_direct(plate_pose):
    """PlatePoseEstimator: angles from synthetic undistorted corner points."""
    alpha_t, beta_t = np.radians(4), np.radians(-3)
    fix_w, cor_w, _, _, _ = synth.scene_geometry(alpha_t, beta_t)
    fix_px = synth.project(plate_pose, fix_w)
    plate_pose.camera_localization(fix_px)
    px = synth.project(plate_pose, cor_w)  # raw fisheye (row, col)
    und = plate_pose.undistort_points(px)
    alpha, beta = plate_pose.estimate_anglesXY(und)
    assert abs(np.degrees(alpha) - 4) < 0.25
    assert abs(np.degrees(beta) + 3) < 0.25


def test_markers_from_csv_path(plate_pose, markers, tmp_path):
    csv = tmp_path / "markers.csv"
    np.savetxt(csv, markers, delimiter=",")
    pipe = EstimationPipeline(fps=55.0, markers=str(csv), estimator="KF")
    frame = synth.render_frame(plate_pose, 0.0, 0.0, ball_m=np.array([0.08, 0.11]))
    x_hat, P, inputs, xb, yb = pipe.estimate(frame)
    assert abs(xb - 0.08) < 0.002


def test_estimators():
    # FiniteDiff: velocity is the mean-based finite difference
    fd = FiniteDiff(55.0, 4)
    meas = np.array([0.1, 0.2])
    x, P = fd.estimate(np.zeros(2), meas)
    assert x.shape == (4,) and np.allclose(x[:2], meas)
    for _ in range(10):
        x, P = fd.estimate(np.zeros(2), meas)
    assert np.allclose(x[2:], 0, atol=1e-12)  # constant position -> zero velocity

    # KF converges to a constant measurement despite nonzero inputs
    kf = KF(55.0)
    z = np.array([0.05, 0.15])
    u = np.array([0.05, -0.05])
    for _ in range(100):
        xm, Pm = kf.estimate(u, z)
    assert np.allclose(xm[:2], z, atol=1e-3)
    assert Pm.shape == (4, 4) and np.all(np.isfinite(Pm))

    # NaN measurement -> prediction only
    xm_pred, P_pred = kf.estimate(u, np.array([np.nan, np.nan]))
    assert np.all(np.isfinite(xm_pred))

    # KFBias runs and estimates a 6-dim state
    kfb = KFBias(55.0)
    xm6, P6 = kfb.estimate(u, z)
    assert xm6.shape == (6,) and P6.shape == (6, 6)
