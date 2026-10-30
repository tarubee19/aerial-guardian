"""Detector benchmark on VisDrone2019-DET-val (548 images, never used for training).

Answers, with numbers instead of claims:
  * How good is the fine-tuned YOLOv8s at finding people (AP50, AP50-95, recall by person size)?
  * How much does fine-tuning on VisDrone buy over off-the-shelf COCO detectors
    (YOLOv8s, YOLOv8m, YOLO11s, RT-DETR-l)?
  * What does imgsz 960 buy over 640 (and would 1280 be better)? -- recall on tiny people + latency.

Ground truth = VisDrone categories 1 (pedestrian) + 2 (people) -> "person", the same mapping
used for training. Ignored regions (category 0, or score flag 0) follow the VisDrone protocol:
detections whose centre falls inside one are discarded before matching.

Usage:
    python benchmarks/detector_benchmark.py --data path/to/VisDrone2019-DET-val --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PERSON_CATS = {1, 2}
# Size buckets on box HEIGHT in original pixels (the README argues in terms of person height).
SIZE_BUCKETS = {"tiny (<16 px)": (0, 16), "small (16-32 px)": (16, 32), "medium (32-64 px)": (32, 64),
                "large (>=64 px)": (64, 1e9)}

CONFIGS = {
    # name: (weights, imgsz, class filter for COCO models)
    "finetuned-yolov8s@640": ("weights/best.pt", 640, None),
    # earlier fine-tune of the nano model (same data and class mapping), kept in the author's Drive
    "finetuned-yolov8n@960": ("weights/yolov8n_visdrone_person.pt", 960, None),
    "finetuned-yolov8s@960": ("weights/best.pt", 960, None),
    "finetuned-yolov8s@1280": ("weights/best.pt", 1280, None),
    "coco-yolov8s@960": ("yolov8s.pt", 960, [0]),
    "coco-yolo11s@960": ("yolo11s.pt", 960, [0]),
    "coco-yolov8m@960": ("yolov8m.pt", 960, [0]),
    "coco-rtdetr-l@960": ("rtdetr-l.pt", 960, [0]),
}


# ----------------------------------------------------------------------------- data
def load_gt(data: Path) -> dict:
    gt = {}
    for img in sorted((data / "images").glob("*.jpg")):
        boxes, ignore = [], []
        for line in (data / "annotations" / f"{img.stem}.txt").read_text().splitlines():
            v = [int(x) for x in line.strip().strip(",").split(",")[:8]]
            x, y, w, h, flag, cat = v[:6]
            if cat == 0 or flag == 0:
                ignore.append([x, y, x + w, y + h])
            elif cat in PERSON_CATS:
                boxes.append([x, y, x + w, y + h])
        gt[img.name] = {"boxes": np.array(boxes, float).reshape(-1, 4),
                        "ignore": np.array(ignore, float).reshape(-1, 4)}
    return gt


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def drop_ignored(det: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    if len(det) == 0 or len(ignore) == 0:
        return det
    cx = (det[:, 0] + det[:, 2]) / 2
    cy = (det[:, 1] + det[:, 3]) / 2
    inside = ((cx[:, None] >= ignore[None, :, 0]) & (cx[:, None] <= ignore[None, :, 2]) &
              (cy[:, None] >= ignore[None, :, 1]) & (cy[:, None] <= ignore[None, :, 3])).any(1)
    return det[~inside]


# ----------------------------------------------------------------------------- matching / AP
def match(dets: dict, gt: dict, iou_thr: float):
    """Greedy COCO-style matching. Returns (scores, is_tp, n_gt, matched_gt_heights_per_score)."""
    scores, tps, gt_hit = [], [], []
    n_gt = 0
    for name, g in gt.items():
        d = dets.get(name, np.zeros((0, 5)))
        d = drop_ignored(d, g["ignore"])
        d = d[np.argsort(-d[:, 4])] if len(d) else d
        gb = g["boxes"]
        n_gt += len(gb)
        used = np.zeros(len(gb), bool)
        ious = iou_matrix(d[:, :4], gb) if len(d) else np.zeros((0, len(gb)))
        for i in range(len(d)):
            j = -1
            if len(gb):
                cand = np.where(~used & (ious[i] >= iou_thr))[0]
                if len(cand):
                    j = cand[np.argmax(ious[i, cand])]
            scores.append(d[i, 4])
            tps.append(j >= 0)
            if j >= 0:
                used[j] = True
    return np.array(scores), np.array(tps, bool), n_gt


def average_precision(scores, tps, n_gt) -> float:
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    o = np.argsort(-scores, kind="stable")
    tp = np.cumsum(tps[o])
    fp = np.cumsum(~tps[o])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def recall_by_size(dets: dict, gt: dict, conf: float, iou_thr: float = 0.5) -> dict:
    hit = {k: 0 for k in SIZE_BUCKETS}
    tot = {k: 0 for k in SIZE_BUCKETS}
    for name, g in gt.items():
        d = dets.get(name, np.zeros((0, 5)))
        d = drop_ignored(d[d[:, 4] >= conf] if len(d) else d, g["ignore"])
        gb = g["boxes"]
        if len(gb) == 0:
            continue
        ious = iou_matrix(gb, d[:, :4]) if len(d) else np.zeros((len(gb), 0))
        found = (ious >= iou_thr).any(1) if ious.shape[1] else np.zeros(len(gb), bool)
        h = gb[:, 3] - gb[:, 1]
        for k, (lo, hi) in SIZE_BUCKETS.items():
            m = (h >= lo) & (h < hi)
            tot[k] += int(m.sum())
            hit[k] += int(found[m].sum())
    return {k: {"recall": hit[k] / tot[k] if tot[k] else None, "n": tot[k]} for k in SIZE_BUCKETS}


def operating_point(dets, gt, conf, iou_thr=0.5) -> dict:
    s, t, n = match({k: v[v[:, 4] >= conf] if len(v) else v for k, v in dets.items()}, gt, iou_thr)
    tp = int(t.sum())
    fp = int((~t).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(n, 1)
    return {"conf": conf, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "tp": tp, "fp": fp,
            "fn": n - tp}


def evaluate(dets: dict, gt: dict, conf_op: float = 0.30) -> dict:
    s, t, n = match(dets, gt, 0.5)
    ap50 = average_precision(s, t, n)
    aps = []
    for thr in np.arange(0.5, 0.96, 0.05):
        s2, t2, _ = match(dets, gt, thr)
        aps.append(average_precision(s2, t2, n))
    best = max((operating_point(dets, gt, c) for c in np.arange(0.05, 0.9, 0.05)), key=lambda r: r["f1"])
    return {"n_gt": n, "AP50": ap50, "AP50_95": float(np.mean(aps)),
            "operating_point": operating_point(dets, gt, conf_op), "best_f1_point": best,
            "recall_by_size@op": recall_by_size(dets, gt, conf_op)}


# ----------------------------------------------------------------------------- inference
def predict(cfg_name: str, data: Path, cache: Path, max_images: int | None = None) -> tuple[dict, dict]:
    from ultralytics import YOLO, RTDETR

    weights, imgsz, classes = CONFIGS[cfg_name]
    f = cache / f"{cfg_name}.json"
    if f.exists():
        raw = json.loads(f.read_text())
        return {k: np.array(v, float).reshape(-1, 5) for k, v in raw["dets"].items()}, raw["timing"]
    model = RTDETR(weights) if "rtdetr" in weights else YOLO(weights)
    imgs = sorted((data / "images").glob("*.jpg"))[:max_images]
    model.predict(str(imgs[0]), imgsz=imgsz, conf=0.01, verbose=False)  # warm-up
    dets, times = {}, []
    for p in imgs:
        t0 = time.perf_counter()
        r = model.predict(str(p), imgsz=imgsz, conf=0.01, iou=0.6, max_det=1000, classes=classes,
                          verbose=False)[0]
        times.append(time.perf_counter() - t0)
        b = r.boxes
        dets[p.name] = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1).round(2).tolist()
    timing = {"device": "cpu", "imgsz": imgsz, "mean_ms": 1000 * float(np.mean(times)),
              "p50_ms": 1000 * float(np.median(times)), "images": len(imgs)}
    cache.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"dets": dets, "timing": timing}))
    return {k: np.array(v, float).reshape(-1, 5) for k, v in dets.items()}, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    a = ap.parse_args()
    gt = load_gt(a.data)
    rows = {}
    for name in a.configs:
        dets, timing = predict(name, a.data, a.out / "raw")
        m = evaluate(dets, gt)
        m["timing"] = timing
        rows[name] = m
        print(f"{name:26s} AP50={m['AP50']:.3f} AP50-95={m['AP50_95']:.3f} "
              f"R@0.30={m['operating_point']['recall']:.3f} P@0.30={m['operating_point']['precision']:.3f} "
              f"{timing['mean_ms']:.0f} ms/img", flush=True)
        (a.out / "detector_benchmark.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
