#!/usr/bin/env python3
"""Compare annotation-seeded tracking methods on an isolated phantom excerpt.

This is research instrumentation, not clinical software or autonomous segmentation.
The next-second annotation is never used to update the preceding prediction.
"""
import argparse
import collections
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
COORDS = re.compile(r"x=([\d.]+),y=([\d.]+)")


def polygon(row):
    if not row:
        return None
    points = [(float(x) * W, float(y) * H) for x, y in COORDS.findall(row["ls_flt_coordinates"])]
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2) if len(points) >= 3 else None


def mask(points, convex=False):
    image = np.zeros((H, W), dtype=np.uint8)
    if points is not None and len(points) >= 3:
        contour = np.rint(points).astype(np.int32)
        if convex:
            contour = cv2.convexHull(contour)
        cv2.fillPoly(image, [contour], 255)
    return image


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else None


def forward_backward(old, new, points):
    target, status, _ = cv2.calcOpticalFlowPyrLK(old, new, points, None, winSize=(25, 25), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, .02))
    if target is None or status is None:
        return None, None
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(new, old, target, None, winSize=(25, 25), maxLevel=3)
    if back is None or back_status is None:
        return None, None
    good = ((status.ravel() == 1) & (back_status.ravel() == 1) &
            (np.linalg.norm(back - points, axis=2).ravel() < 3) &
            (target[:, 0, 0] >= 0) & (target[:, 0, 0] < W) &
            (target[:, 0, 1] >= 0) & (target[:, 0, 1] < H))
    return points[good], target[good]


def features(gray, boundary):
    if boundary is None:
        return None
    # Features are selected only inside the current estimated contour.
    return cv2.goodFeaturesToTrack(gray, maxCorners=120, qualityLevel=.01, minDistance=4,
                                   mask=mask(boundary), blockSize=7)


def run(video, rows, method, output):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError("Cannot open video or determine frame rate")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps / 2, (W, H))
    if not writer.isOpened():
        raise RuntimeError("Cannot open output video writer")
    previous = boundary = tracked = None
    last_second = -1
    counts = collections.Counter()
    checks = []
    index = 0
    last_rendered = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = index
            index += 1
            if frame_index % 2:
                continue
            t = frame_index / fps
            if t >= 60:
                break
            sec = int(t)
            frame = cv2.resize(frame, (W, H))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            row = rows.get(sec)
            gt = polygon(row)
            anchor = sec != last_second
            if anchor:
                # Score the preceding frame prediction before reading this new anchor.
                prev_row = rows.get(sec - 1)
                if last_second >= 0 and gt is not None and polygon(prev_row) is not None:
                    predicted = mask(boundary, convex=(method == "baseline"))
                    target = mask(gt)
                    checks.append({"target_second": sec, "prediction_time": last_rendered,
                                   "had_prediction": bool(predicted.any()),
                                   "iou": iou(predicted, target)})
                boundary = gt.copy() if gt is not None else None
                tracked = boundary.copy() if method == "baseline" and boundary is not None else features(gray, boundary)
                last_second = sec
            elif boundary is not None and previous is not None and tracked is not None:
                old_points, new_points = forward_backward(previous, gray, tracked)
                if old_points is None or len(old_points) < 3:
                    boundary = tracked = None
                    counts["lost_too_few_points"] += 1
                elif method == "baseline":
                    boundary = tracked = new_points.reshape(-1, 1, 2)
                else:
                    matrix, inliers = cv2.estimateAffinePartial2D(old_points.reshape(-1, 2), new_points.reshape(-1, 2),
                                                                 method=cv2.RANSAC, ransacReprojThreshold=3)
                    if matrix is None or inliers is None or int(inliers.sum()) < 3:
                        boundary = tracked = None
                        counts["lost_affine"] += 1
                    else:
                        boundary = cv2.transform(boundary, matrix)
                        tracked = features(gray, boundary)
                        if tracked is None or len(tracked) < 3:
                            boundary = tracked = None
                            counts["lost_no_features"] += 1
            elif not anchor:
                boundary = tracked = None
            valid = boundary is not None and len(boundary) >= 3
            counts["frames"] += 1
            counts["visible_annotation_frames"] += int(gt is not None)
            counts["overlay_frames"] += int(valid)
            counts["visible_annotation_overlay_frames"] += int(valid and gt is not None)
            counts["anchor_frames"] += int(anchor)
            if valid:
                p = np.rint(boundary).astype(np.int32)
                shape = cv2.convexHull(p) if method == "baseline" else p
                overlay = frame.copy()
                cv2.fillPoly(overlay, [shape], (215, 208, 65))
                frame = cv2.addWeighted(frame, .8, overlay, .2, 0)
                cv2.polylines(frame, [shape], True, (215, 208, 65), 2)
            state = "ANCHOR" if anchor else ("TRACKED" if valid else "TRACK LOST")
            label = row["str_instrument"].replace("_", " ") if row else "no annotation"
            cv2.rectangle(frame, (0, 0), (W, 52), (15, 23, 28), -1)
            cv2.putText(frame, f"CASE 11 | {method.upper()} | {t:05.2f}s | {label.upper()}",
                        (12, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (225, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{state} | ANNOTATION-SEEDED | NOT AUTONOMOUS SEGMENTATION",
                        (12, 43), cv2.FONT_HERSHEY_SIMPLEX, .42, (225, 235, 235), 1, cv2.LINE_AA)
            writer.write(frame)
            previous = gray
            last_rendered = t
    finally:
        cap.release()
        writer.release()
    return {"method": method, "fps_input": fps, "frames": dict(counts), "next_anchor_checks": checks,
            "next_anchor_mean_iou_including_loss": float(np.mean([x["iou"] for x in checks])) if checks else None,
            "next_anchor_prediction_rate": float(np.mean([x["had_prediction"] for x in checks])) if checks else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.annotations.open(newline="", encoding="utf-8") as f:
        rows = {int(r["int_second"]): r for r in csv.DictReader(f) if r["int_video"] == "11"}
    if any(i not in rows for i in range(60)):
        raise ValueError("Missing case 11 annotations in the first minute")
    results = [run(args.video, rows, method, args.out / f"case11-{method}-60s.mp4")
               for method in ("baseline", "affine_features")]
    (args.out / "comparison.json").write_text(json.dumps(results, indent=2))
    for r in results:
        print(r["method"], r["frames"], "next-anchor checks", len(r["next_anchor_checks"]),
              "mean IoU", r["next_anchor_mean_iou_including_loss"])


if __name__ == "__main__":
    main()
