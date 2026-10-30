"""Does each per-frame filter in the stack earn its place?  (VisDrone2019-DET-val, 548 images)

The tracker applies four *per-frame* filters to YOLO's boxes before the temporal ones
(static filter, confirmation gate), which need video and are measured in the tracking eval:

  1. adaptive confidence gate (small boxes need higher confidence)
  2. geometry filter (aspect ratio, min size, max area)
  3. cross-class NMS (IoU > 0.25 -> keep the larger box)
  4. merged-box suppression (a box containing 2+ other boxes is dropped)

Here every filter is switched on/off on the same cached detections of the fine-tuned model
(detector_benchmark.py must have run first), using the exact methods from src/tracker.py and the
scene profile the tracker would choose for each image. Reported: precision, recall, F1 and the
number of true people lost, at IoU 0.5 against pedestrian + people boxes.

Usage:  python benchmarks/filter_ablation.py --data path/to/VisDrone2019-DET-val
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector_benchmark import load_gt, match, recall_by_size  # noqa: E402
from tracker import DronePersonTracker  # noqa: E402

FILTERS = ("adaptive_conf", "geometry", "cross_class_nms", "merged_box")


def make_filterer() -> DronePersonTracker:
    t = DronePersonTracker.__new__(DronePersonTracker)   # no model needed, only the filter methods
    t.p = DronePersonTracker.SCENE_PROFILES["day"].copy()
    return t


def apply(t: DronePersonTracker, det: np.ndarray, scene: str, on: set[str]) -> np.ndarray:
    t.p = DronePersonTracker.SCENE_PROFILES[scene].copy()
    # same pre-filter confidence the tracker passes to model.track()
    det = det[det[:, 4] >= max(0.18, t.p["conf_base"] - 0.12)]
    boxes, confs = det[:, :4], det[:, 4]
    ids = np.arange(len(det))
    cls = np.zeros(len(det), int)
    if "cross_class_nms" in on:
        boxes, ids, confs, cls = t._cross_class_nms(boxes, ids, confs, cls)
    if "merged_box" in on:
        boxes, ids, confs, cls = t._kill_merged_boxes(boxes, ids, confs, cls)
    keep = []
    for b, c in zip(boxes, confs):
        if "adaptive_conf" in on and c < t._adaptive_conf(*b):
            keep.append(False)
            continue
        if "geometry" in on and not t._geometry_ok(*b):
            keep.append(False)
            continue
        keep.append(True)
    keep = np.array(keep, bool)
    return np.c_[boxes[keep], confs[keep]] if keep.any() else np.zeros((0, 5))


def prf(dets: dict, gt: dict) -> dict:
    s, t, n = match(dets, gt, 0.5)
    tp, fp = int(t.sum()), int((~t).sum())
    p, r = tp / max(tp + fp, 1), tp / max(n, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "tp": tp, "fp": fp, "fn": n - tp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=Path("benchmarks/results/raw/finetuned-yolov8s@960.json"))
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results/filter_ablation.json"))
    a = ap.parse_args()
    gt = load_gt(a.data)
    raw = {k: np.array(v, float).reshape(-1, 5) for k, v in json.loads(a.raw.read_text())["dets"].items()}
    t = make_filterer()
    scenes = {name: t._auto_detect_scene(cv2.imread(str(a.data / "images" / name))) for name in gt}
    counts = {s: list(scenes.values()).count(s) for s in set(scenes.values())}
    print("scene profile chosen per image:", counts)
    results = {"scenes": counts, "configs": {}}
    configs = {"no filters": set(), "full stack": set(FILTERS)}
    for f in FILTERS:
        configs[f"only {f}"] = {f}
        configs[f"full stack minus {f}"] = set(FILTERS) - {f}
    for name, on in configs.items():
        dets = {k: apply(t, v, scenes[k], on) for k, v in raw.items()}
        r = prf(dets, gt)
        r["recall_by_size"] = recall_by_size(dets, gt, 0.0)
        results["configs"][name] = r
        print(f"{name:34s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} FP={r['fp']} FN={r['fn']}")
    a.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
