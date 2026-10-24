#!/usr/bin/env python3
"""
Interactive marker selection tool for crs2r-image-processing.

Click 8 points on the image in order:
  1-4  Outer fixed dots   (corners of the outer frame, any consistent order)
  5-8  Maze corner dots   (corners of the maze playing field, same order)

Output: markers.csv  —  8 rows of "col,row" (u,v convention),
        ready to pass to EstimationPipeline(markers="markers.csv") or
        np.loadtxt("markers.csv", delimiter=",").

Usage:
    python select_markers.py <image_path> [--output markers.csv] [--scale 0.5]

Controls:
    Left-click  — place next marker
    Right-click — undo last marker
    Enter / s   — save and quit (only when all 8 points are placed)
    Escape / q  — quit without saving
"""

import argparse
import sys
import cv2
import numpy as np

# ── Labels shown while clicking ──────────────────────────────────────────────
LABELS = [
    "1 fixed",
    "2 fixed",
    "3 fixed",
    "4 fixed",
    "5 maze",
    "6 maze",
    "7 maze",
    "8 maze",
]

# Colours: fixed dots → blue, maze corners → green
COLOURS = [(255, 80, 0)] * 4 + [(0, 200, 60)] * 4
N_MARKERS = 8


def draw_state(base: np.ndarray, points: list[tuple[int, int]], scale: float) -> np.ndarray:
    vis = base.copy()
    h, w = vis.shape[:2]

    # Separator line between the two groups
    sep_y = 30
    cv2.rectangle(vis, (0, 0), (w, sep_y + 8), (30, 30, 30), -1)

    if len(points) < N_MARKERS:
        status = f"Click point {len(points) + 1}/8 — {LABELS[len(points)]}"
        status_col = (230, 230, 230)
    else:
        status = "All 8 markers placed — press Enter/s to save, Right-click to undo"
        status_col = (80, 255, 80)

    cv2.putText(vis, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 1, cv2.LINE_AA)

    for i, (cx, cy) in enumerate(points):
        colour = COLOURS[i]
        cv2.drawMarker(vis, (cx, cy), colour, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        label = LABELS[i]
        cv2.putText(vis, label, (cx + 8, cy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colour, 1, cv2.LINE_AA)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Select 8 markers on an image.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("--output", default="markers.csv",
                        help="Output CSV path (default: markers.csv).")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Display scale factor, e.g. 0.5 to halve (default: 1.0).")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Error: cannot open image '{args.image}'")

    scale = args.scale
    if scale != 1.0:
        disp = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        disp = img.copy()

    points: list[tuple[int, int]] = []  # display-space (col, row)

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < N_MARKERS:
                points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()

    win = "Select markers  (Left-click=place  Right-click=undo  Enter/s=save  Esc/q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        frame = draw_state(disp, points, scale)
        cv2.imshow(win, frame)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):  # Esc / q → quit without saving
            print("Aborted — no file written.")
            break

        if key in (13, ord("s")):  # Enter / s → save
            if len(points) < N_MARKERS:
                print(f"Need {N_MARKERS} points, only {len(points)} placed — keep clicking.")
                continue
            # Convert from display-space back to original image space,
            # then store as (col, row) = (u, v) as required by EstimationPipeline.
            pts_orig = np.array(points, dtype=float)
            pts_orig /= scale  # undo display scaling
            np.savetxt(args.output, pts_orig, delimiter=",", fmt="%.4f")
            print(f"Saved {N_MARKERS} markers to '{args.output}'")
            print("  Row order: 4 fixed dots, then 4 maze-corner dots")
            print("  Column format: col (u), row (v)")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
