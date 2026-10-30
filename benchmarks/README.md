# Benchmarks

Three scripts turn the design claims in the main README into measurements. All of them use the exact filter
methods in `src/tracker.py`, so what is measured is what ships.

| Script | Data | Measures | Runtime |
|---|---|---|---|
| `detector_benchmark.py` | VisDrone2019-DET-val (548 images) | AP50, AP50-95, precision/recall at a threshold, recall by person height, latency | ~5 min per config on CPU |
| `filter_ablation.py` | same, reuses cached detections | precision / recall / F1 with every per-frame filter on or off | ~2 min |
| `tracking_eval.py` | VisDrone2019-MOT-val (7 sequences) | MOTA, IDF1, ID switches, FP, FN, mostly tracked/lost, FPS | ~5-10 min per config on a T4 |

```bash
pip install -r requirements.txt motmetrics
# DET-val is mirrored by ultralytics on GitHub (VisDrone licence applies)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip && unzip VisDrone2019-DET-val.zip
python benchmarks/detector_benchmark.py --data VisDrone2019-DET-val
python benchmarks/filter_ablation.py   --data VisDrone2019-DET-val
# MOT-val: download from the VisDrone GitHub page, then
python benchmarks/tracking_eval.py --mot path/to/VisDrone2019-MOT-val --model weights/best.pt
```

## Protocol decisions (and why)

* **Ground truth = `pedestrian` + `people`.** That is the mapping the detector was trained with; scoring it on
  cyclists or cars would measure something it was never asked to do.
* **Ignored regions** (category 0, or the "score" flag 0, plus MOT category 11 "others") are handled as in the
  VisDrone toolkit: a prediction whose centre falls inside one is removed before matching, so a correct detection
  in an unlabelled crowd is not counted as a false positive.
* **AP** uses greedy matching at IoU 0.5 (and 0.50:0.95 in 0.05 steps) with 101-point interpolation, like COCO.
  Detections are produced at confidence 0.01 so the full precision-recall curve is available; the operating point
  (0.30) is the best-F1 threshold for the fine-tuned model.
* **Size buckets use box height**, because the README argues about "an 8-40 px tall person".
* **COCO models** are restricted to their `person` class; their numbers show what fine-tuning buys, not a tuned
  competitor.
* **Latency** is measured on whatever runs the script (a 2-vCPU cloud machine for the committed numbers). Use it to
  compare configurations, not as T4 or Jetson FPS.
* **Tracking metrics** come from py-motmetrics at IoU 0.5 (its IoU helper uses an API removed in NumPy 2, so the
  script computes the distance matrix itself). Only confirmed tracks are output by the pipeline, exactly what the
  video overlay shows.

## Findings worth knowing before an interview

1. The shipped checkpoint is **single-class**. The person/rider logic (vehicle suppression, cross-class NMS) was
   written for an earlier two-class model; with this checkpoint cross-class NMS is a class-agnostic NMS at IoU 0.25
   that removes adjacent people in crowds.
2. **ReID had no effect**: matching a new track to a lost one deleted the lost entry but never re-assigned the id.
   `reid_link` (opt-in) fixes it so that its value can be measured.
3. The model file is committed as `weights/best(8).pt` while every command uses `weights/best.pt`; copy it
   (`cp "weights/best(8).pt" weights/best.pt`). The workflow does this automatically.
