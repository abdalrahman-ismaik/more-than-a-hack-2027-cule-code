"""Ball detection in the tactical video, mapped to pitch coordinates (optional evidence).

The tracking data loses the ball on 43% of frames. This module runs a pretrained ball detector on
the tactical video and maps the best detection in each frame to the pitch, using the homography.

Detector: martinjolif/yolo-football-ball-detection (YOLO11n, AGPL-3.0). Weights are in models/.
Homography: one 3x3 matrix per frame, mapping normalised pitch coordinates to image pixels, so the
inverse maps a pixel back to the pitch (DATA.md, "Homography").

Detections are kept only if they land on the pitch (with a margin), because the detector also
fires on players' feet and on the crowd.

    python -m src.video_ball --start 11185 --stop 11485 --out data/video_ball_test.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.config import N_FRAMES, ROOT, TACTICAL_DIR

WEIGHTS = os.path.join(ROOT, "models", "yolo-football-ball-detection.pt")
VIDEO = os.path.join(TACTICAL_DIR, "BAR-ATH_tactical.mp4")
HOMOGRAPHY = os.path.join(TACTICAL_DIR, "BAR-ATH_tactical_homography.csv")
CONF = 0.1
IMGSZ = 1280
PITCH_MARGIN = 0.02  # accept detections up to 2% of the pitch outside the lines


def load_homography(path=HOMOGRAPHY):
    """(H, valid): H is (n, 3, 3) with NaN where invalid; valid is a boolean array."""
    table = pd.read_csv(path)
    cols = ["h%d%d" % (r, c) for r in range(3) for c in range(3)]
    H = table[cols].to_numpy(np.float64).reshape(-1, 3, 3)
    valid = table["valid"].to_numpy(bool)
    return H, valid


def pixel_to_pitch(H, px, py):
    """Map image pixels to normalised pitch coordinates using the inverse of each frame's H."""
    inv = np.linalg.inv(H)
    p = np.stack([px, py, np.ones_like(px)], axis=-1)
    q = np.einsum("nij,nj->ni", inv, p)
    return q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]


def detect_range(start, stop, device=None, out=None, batch=8):
    """Run the detector on frames [start, stop) and return one row per frame."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(WEIGHTS)
    H, valid = load_homography()
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    rows = []
    frame_id = start
    while frame_id < stop:
        frames, ids = [], []
        while len(frames) < batch and frame_id < stop:
            ok, image = cap.read()
            if not ok:
                frame_id = stop
                break
            frames.append(image)
            ids.append(frame_id)
            frame_id += 1
        if not frames:
            break
        results = model.predict(frames, conf=CONF, imgsz=IMGSZ, verbose=False,
                                device=device, classes=[0])
        for index, result in zip(ids, results):
            rows.append(_best_on_pitch(index, result, H, valid))

    cap.release()
    table = pd.DataFrame(rows)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        table.to_csv(out, index=False)
    return table


def _best_on_pitch(index, result, H, valid):
    """Highest-confidence detection whose pitch position is on the pitch, or empty."""
    empty = {"frame": index, "det_x": np.nan, "det_y": np.nan, "det_conf": np.nan,
             "n_boxes": 0, "h_valid": bool(valid[index])}
    if result.boxes is None or len(result.boxes) == 0 or not valid[index]:
        return empty
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    x, y = pixel_to_pitch(np.repeat(H[index][None], len(cx), axis=0), cx, cy)
    on = (x >= -PITCH_MARGIN) & (x <= 1 + PITCH_MARGIN) & (y >= -PITCH_MARGIN) & (y <= 1 + PITCH_MARGIN)
    if not on.any():
        return dict(empty, n_boxes=len(cx))
    best = np.argmax(np.where(on, confs, -1))
    return {"frame": index, "det_x": float(x[best]), "det_y": float(y[best]),
            "det_conf": float(confs[best]), "n_boxes": int(len(cx)), "h_valid": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=N_FRAMES)
    parser.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu for CPU")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    table = detect_range(args.start, args.stop, device=args.device, out=args.out)
    print("frames {}  with a pitch detection {}  ({:.1f}%)".format(
        len(table), int(table["det_x"].notna().sum()), 100 * table["det_x"].notna().mean()))
    print("wrote", args.out)
