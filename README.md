# crs2r-image-processing

Lean, ROS-free port of the state estimation module from `cyberrunner_state_estimation`,
kept as close to the original implementation as possible. Dependencies: numpy, opencv-python, scipy.

```python
from crs2r_image_processing.state_estimation import EstimationPipeline

pipeline = EstimationPipeline(
    fps=55.0,
    markers=markers,        # (8, 2) array or path to markers.csv:
                            # 4 outer fixed dots, then 4 maze corner dots,
                            # rows in (u, v) = (col, row) — as produced by the
                            # original select_markers tool
    estimator="KF",         # "FiniteDiff" | "KF" | "KFBias"
)

x_hat, P, (alpha, beta), xb, yb = pipeline.estimate(frame)          # frame: (400, 640, 3) BGR
x_hat, P, (alpha, beta), subimg, xb, yb = pipeline.estimate(frame, return_ball_subimg=True)
```

## High-level API (same outputs as the original ROS module)

- `EstimationPipeline.estimate(frame)` — full per-frame pipeline: mask, corner
  and ball detection, fisheye undistortion of the points, plate pose, ball
  back-projection, state estimation.
- `Measurements` — the measurement layer on its own:
  - `process_frame(frame, get_ball_subimg=False)`
  - `get_ball_position_in_maze()` → ball (x, y, z) in the maze frame {m};
    since {m} is attached to the (tilted) plate, (x, y) is already normalized
    with respect to the plate angle (untwisted).
  - `get_plate_pose()` → (alpha, beta) in rad around +X / +Y.
  - `get_ball_coordinates()` → raw fisheye pixel coords of the ball.
  - `get_ball_subimg()` → 64x64 subimage centered on the ball, sampled on the
    tilted plate plane over +/- 3.2 cm (the original `1e-3 * mgrid[-32:32]`
    range), mapped back through the fisheye model.
  - `extract_ball_subimg(frame, ball_pos)` — the subimage extraction on its own.

## Lower-level API

- `OcamModel` (from the original ocam port):
  - `undistort(img)` — un-fisheye a full image (calibrated 1920x1200 model).
  - `scale(factor)` — scale the model to a lower resolution (the pipeline uses
    factor 3: 1920x1200 → 640x400).
  - `cam2world` / `world2cam` — pixel ↔ ray projection.
- `PlatePoseEstimator`:
  - `undistort_points(pts)` — fisheye → pinhole (K) coordinates.
  - `estimate_anglesXY(corners_undist)` — untwist: plate angles from the
    undistorted corner dots.
  - `camera_localization(fix_pts)` — camera pose from the 4 outer fixed dots.
- `Detector` / `DetectorFixedPts` — HSV blob detection of corner dots and ball.
- `KF`, `KFBias`, `FiniteDiff` — state estimators (unchanged).

The camera calibration file is bundled at
`crs2r_image_processing/state_estimation/data/calib_results_cyberrunner.txt`.

## Not ported (ROS/matplotlib-specific)

ROS node/publishers, `anim_3d`/`camera_pose_visualizer` (matplotlib), the
profile tools, and the interactive marker-selection GUI. `markers.csv` must be
supplied by the caller.

## Tests

`uv run pytest` — validated against a synthetic fisheye scene rendered through
the real calibration model (exact ground truth for angles and ball position):
`tests/synth.py` builds the scene, `tests/test_state_estimation.py` checks the
ocam round-trip, un-fisheye mapping, plate-angle recovery (<0.25 deg error at
4 deg tilt), ball position (<2 mm), 64x64 subimage centering/orientation,
masking, and the estimators.
