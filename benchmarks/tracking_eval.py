"""Multi-object tracking evaluation on VisDrone2019-MOT-val (7 sequences, ~2,800 frames).

The original project showed tracked videos but no tracking metric. This script scores every
tracker configuration against the ground truth with the CLEAR-MOT and identity metrics
(py-motmetrics): MOTA, IDF1, ID switches, false positives/negatives, mostly tracked/lost.

Ground truth: VisDrone MOT categories 1 (pedestrian) + 2 (people) = "person", the classes the
detector was trained on. Ignored regions (category 0) and "others" (11) follow the VisDrone
protocol: predictions whose centre falls inside one are removed before matching. IoU >= 0.5.

Configurations (all with the fine-tuned YOLOv8s at imgsz 960):
  bytetrack            ultralytics ByteTrack defaults
  botsort-no-gmc       ultralytics BoT-SORT defaults with camera-motion compensation OFF
  botsort-default      ultralytics BoT-SORT defaults (GMC = sparseOptFlow, the ultralytics default)
  botsort-drone        the repo's tuned configs/drone_botsort.yaml, raw tracker output
  pipeline-full        DronePersonTracker with the full filter stack (the shipped system)
  pipeline-minus-X     the same with one filter switched off (ablation)
  pipeline-improved    filters kept only where the evidence supports them + ReID id-linking

Usage (GPU strongly recommended):
  python benchmarks/tracking_eval.py --mot path/to/VisDrone2019-MOT-val --model weights/best.pt \
      --out benchmarks/results/tracking [--configs ...] [--max-frames N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PERSON = {1, 2}
IGNORE = {0, 11}

FULL = {"vehicle", "cross_class_nms", "merged_box", "adaptive_conf", "geometry", "static", "confirm", "reid"}
CONFIGS = {
    "bytetrack": ("raw", "bytetrack.yaml", None),
    "botsort-no-gmc": ("raw", "botsort_nogmc", None),
    "botsort-default": ("raw", "botsort.yaml", None),
    "botsort-drone": ("raw", str(ROOT / "configs" / "drone_botsort.yaml"), None),
    "pipeline-full": ("pipeline", str(ROOT / "configs" / "drone_botsort.yaml"), FULL),
    **{f"pipeline-minus-{f}": ("pipeline", str(ROOT / "configs" / "drone_botsort.yaml"), FULL - {f})
       for f in ("cross_class_nms", "adaptive_conf", "geometry", "merged_box", "static", "confirm")},
    "pipeline-improved": ("pipeline", str(ROOT / "configs" / "drone_botsort.yaml"),
                          {"adaptive_conf", "merged_box", "confirm", "reid", "reid_link"}),
}


def load_gt(ann: Path) -> tuple[dict, dict]:
    a = pd.read_csv(ann, header=None).iloc[:, :8]
    a.columns = ["frame", "id", "x", "y", "w", "h", "score", "cat"]
    gt, ign = {}, {}
    for f, g in a.groupby("frame"):
        p = g[g.cat.isin(PERSON) & (g.score != 0)]
        gt[int(f)] = (p.id.to_numpy(), p[["x", "y", "w", "h"]].to_numpy(float))
        i = g[g.cat.isin(IGNORE) | (g.score == 0)]
        ign[int(f)] = i[["x", "y", "w", "h"]].to_numpy(float)
    return gt, ign


def drop_ignored(boxes: np.ndarray, ign: np.ndarray) -> np.ndarray:
    if len(boxes) == 0 or len(ign) == 0:
        return np.ones(len(boxes), bool)
    cx = boxes[:, 0] + boxes[:, 2] / 2
    cy = boxes[:, 1] + boxes[:, 3] / 2
    inside = ((cx[:, None] >= ign[None, :, 0]) & (cx[:, None] <= ign[None, :, 0] + ign[None, :, 2]) &
              (cy[:, None] >= ign[None, :, 1]) & (cy[:, None] <= ign[None, :, 1] + ign[None, :, 3])).any(1)
    return ~inside


def nogmc_yaml(tmp: Path) -> str:
    import ultralytics
    src = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml"
    text = src.read_text()
    text = "\n".join("gmc_method: none" if line.startswith("gmc_method") else line for line in text.splitlines())
    out = tmp / "botsort_nogmc.yaml"
    out.write_text(text)
    return str(out)


def run_sequence(kind, tracker_yaml, filters, model_path, frames, imgsz):
    """Returns {frame: [(id, x, y, w, h)]} and seconds per frame."""
    import cv2
    out, t_total = {}, 0.0
    if kind == "raw":
        from ultralytics import YOLO
        model = YOLO(model_path)
        for fi, path in enumerate(frames, 1):
            img = cv2.imread(str(path))
            t0 = time.perf_counter()
            r = model.track(img, persist=True, tracker=tracker_yaml, imgsz=imgsz, verbose=False)[0]
            t_total += time.perf_counter() - t0
            rows = []
            if r.boxes.id is not None:
                for tid, (x1, y1, x2, y2) in zip(r.boxes.id.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                    rows.append((int(tid), x1, y1, x2 - x1, y2 - y1))
            out[fi] = rows
    else:
        from tracker import DronePersonTracker
        trk = DronePersonTracker(model_path, tracker_yaml, filters=filters, imgsz=imgsz, draw=False)
        for fi, path in enumerate(frames, 1):
            img = cv2.imread(str(path))
            t0 = time.perf_counter()
            trk.process_frame(img)
            t_total += time.perf_counter() - t0
            out[fi] = [(tid, x1, y1, x2 - x1, y2 - y1) for tid, x1, y1, x2, y2, _ in trk.last_tracks]
    return out, t_total / max(len(frames), 1)


def iou_distance(a: np.ndarray, b: np.ndarray, min_iou: float = 0.5) -> np.ndarray:
    """1 - IoU for xywh boxes, NaN (= cannot match) below min_iou. (motmetrics' own helper
    uses np.asfarray, removed in NumPy 2.)"""
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)))
    a2, b2 = a.copy(), b.copy()
    a2[:, 2:] += a2[:, :2]
    b2[:, 2:] += b2[:, :2]
    x1 = np.maximum(a2[:, None, 0], b2[None, :, 0])
    y1 = np.maximum(a2[:, None, 1], b2[None, :, 1])
    x2 = np.minimum(a2[:, None, 2], b2[None, :, 2])
    y2 = np.minimum(a2[:, None, 3], b2[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    iou = inter / (a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter + 1e-9)
    d = 1 - iou
    d[iou < min_iou] = np.nan
    return d


def score(pred: dict, gt: dict, ign: dict, name: str):
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=False)
    for f in sorted(gt):
        g_ids, g_boxes = gt[f]
        p = pred.get(f, [])
        p_ids = np.array([r[0] for r in p])
        p_boxes = np.array([r[1:] for r in p], float).reshape(-1, 4)
        keep = drop_ignored(p_boxes, ign.get(f, np.zeros((0, 4))))
        p_ids, p_boxes = p_ids[keep] if len(p_ids) else p_ids, p_boxes[keep]
        d = iou_distance(g_boxes, p_boxes)
        acc.update(g_ids.tolist(), p_ids.tolist(), d, frameid=f)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot", type=Path, required=True, help="VisDrone2019-MOT-val folder (sequences/, annotations/)")
    ap.add_argument("--model", default=str(ROOT / "weights" / "best.pt"))
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results" / "tracking")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--max-frames", type=int, default=0, help="limit frames per sequence (0 = all)")
    a = ap.parse_args()
    import motmetrics as mm
    a.out.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p for p in (a.mot / "sequences").iterdir() if p.is_dir())
    gts = {s.name: load_gt(a.mot / "annotations" / f"{s.name}.txt") for s in seqs}
    summary_rows = {}
    for cname in a.configs:
        kind, yaml_path, filters = CONFIGS[cname]
        if yaml_path == "botsort_nogmc":
            yaml_path = nogmc_yaml(a.out)
        cache = a.out / f"{cname}.json"
        if cache.exists():
            saved = json.loads(cache.read_text())
            preds = {k: {int(f): v for f, v in d.items()} for k, d in saved["preds"].items()}
            spf = saved["sec_per_frame"]
        else:
            preds, spf = {}, {}
            for s in seqs:
                frames = sorted(s.glob("*.jpg"))
                if a.max_frames:
                    frames = frames[: a.max_frames]
                preds[s.name], spf[s.name] = run_sequence(kind, yaml_path, filters, a.model, frames, a.imgsz)
                print(f"  {cname:30s} {s.name} {len(frames)} frames {1 / max(spf[s.name], 1e-9):.1f} FPS", flush=True)
            cache.write_text(json.dumps({"preds": {k: {str(f): v for f, v in d.items()} for k, d in preds.items()},
                                         "sec_per_frame": spf}, default=float))
        accs, names = [], []
        for s in seqs:
            gt, ign = gts[s.name]
            if a.max_frames:
                gt = {f: v for f, v in gt.items() if f <= a.max_frames}
            accs.append(score(preds[s.name], gt, ign, s.name))
            names.append(s.name)
        mh = mm.metrics.create()
        metrics = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses", "mostly_tracked",
                   "mostly_lost", "num_unique_objects", "precision", "recall"]
        df = mh.compute_many(accs, metrics=metrics, names=names, generate_overall=True)
        df["fps"] = [1 / max(spf[n], 1e-9) for n in names] + [len(spf) / sum(spf.values())]
        df.to_csv(a.out / f"{cname}_per_sequence.csv")
        summary_rows[cname] = df.loc["OVERALL"].to_dict()
        o = summary_rows[cname]
        print(f"{cname:30s} MOTA={o['mota']:.3f} IDF1={o['idf1']:.3f} IDSW={int(o['num_switches'])} "
              f"FP={int(o['num_false_positives'])} FN={int(o['num_misses'])}", flush=True)
        pd.DataFrame(summary_rows).T.to_csv(a.out / "summary.csv")


if __name__ == "__main__":
    main()
