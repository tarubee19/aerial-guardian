# aerial-guardian
# Aerial Guardian — Drone Person Detection & Tracking

Multi-object tracking pipeline for persons in drone footage using YOLOv8n fine-tuned on VisDrone2019, SAHI sliced inference, and ByteTrack MOT.

---

## Quick start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/aerial-guardian.git
cd aerial-guardian
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the model
Download `best.pt` (6.3 MB) from:
[Google Drive link — add your link here]

Place it at `models/best.pt`.

### 4. Download the dataset
VisDrone2019-MOT-val from:
https://github.com/VisDrone/VisDrone-Dataset

Place extracted folder at `data/VisDrone2019-MOT-val/`.

### 5. Run inference
Open `inference/pipeline.ipynb` in Google Colab or Jupyter.
Run all cells in order. Output MP4 clips saved to `outputs/`.

---

## Hardware used for testing
- GPU: NVIDIA Tesla T4 (Google Colab)
- VRAM: 15.6 GB
- Average FPS: 1.2–4.0 depending on sequence density
- Model size: 6.3 MB

---

## Summary Report

### Architecture choice and small object detection

**Base model: YOLOv8n (nano)**
YOLOv8n was chosen as the detector backbone for three reasons: (1) it is 6.3 MB — well within the 300 MB constraint; (2) it runs at 3.1 ms inference per frame on a T4 GPU; (3) its multi-scale feature pyramid (P3/8, P4/16, P5/32) detects objects down to 8×8 pixels, which covers the smallest persons in VisDrone (~15×30px).

The model was fine-tuned from YOLOv8n pretrained weights on the VisDrone2019 training set for 50 epochs at 960px resolution, reducing from the default 80-class COCO head to a 2-class head (pedestrian, people). Final validation mAP50 = 0.481, mAP50-95 = 0.202.

**SAHI (Slicing Aided Hyper Inference)**
Raw VisDrone frames are 1920×1080. Running the detector on the full resolution misses small persons because they occupy only ~0.1% of frame pixels. SAHI tiles each frame into overlapping 512×512 crops (25% overlap), runs inference on each tile independently, then merges detections back into full-frame coordinates using NMS. This effectively gives the model a 3–4× resolution advantage on small objects with no architectural changes. The tile size of 512px was chosen to match the model's training resolution of 960px at roughly 0.5× scale, keeping persons at a detectable apparent size.

**Drone-specific additions beyond the base model:**
- Aspect ratio filter (h/w 0.5–4.5): removes vehicle detections which are always wider than tall
- Minimum width filter (14–20px depending on scene): removes pole and sign detections which are narrower than any human shoulder width
- Scene-type parameter switching: night sequences use higher confidence threshold (0.38 vs 0.28) because illuminated objects produce low-confidence false positives that real persons exceed

### ID switching and ego-motion

Drone footage has two sources of ID switching that ground-camera trackers do not face:

**1. Ego-motion (camera translation/rotation)**
When the drone moves, all objects shift in image coordinates simultaneously. ByteTrack's Kalman filter assumes objects move independently — it cannot predict a global frame shift. This causes the predicted bounding box positions to diverge from actual detections, IoU drops below the matching threshold, and new IDs are assigned.

Fix applied: ORB feature matching between consecutive frames estimates a homography matrix representing the camera's frame-to-frame motion. This compensates for global frame shifts before the Kalman prediction step, so predicted positions remain accurate even during drone translation. 500 ORB keypoints are extracted per frame; the top 100 matches are used with RANSAC to robustly estimate the homography even when foreground objects (persons) are moving.

**2. Occlusion and re-appearance**
When persons pass behind vehicles or each other, they disappear for several frames. ByteTrack's `track_buffer=60` (4 seconds at 15fps) keeps the Kalman state alive during absence. `match_thresh=0.60` allows re-association at lower IoU, tolerating the position uncertainty that accumulates while a track is lost.

ByteTrack was chosen over DeepSORT specifically because DeepSORT requires a re-identification embedding model (~50–150 MB) which would bring total model size over the 300 MB constraint and reduce FPS. ByteTrack is purely geometric — IoU + Kalman — adding zero model weight and negligible compute overhead (~0.5 ms per frame).

**3. Static object suppression**
A `StaticObjectFilter` tracks the spread of each ByteTrack ID's centroid history over the last 8 frames. Any track whose centroid moves less than 6 pixels total is suppressed from the output — it is a stationary object (sign, pole, parked vehicle) not a person. Real pedestrians always displace more than 6 pixels per 8 frames at drone altitudes.

### Edge deployment on NVIDIA Jetson

To deploy on Jetson Orin Nano (8 GB, the current recommended edge platform):

**Step 1 — Export to TensorRT**
```bash
yolo export model=best.pt format=engine device=0 half=True
```
This converts the PyTorch model to a TensorRT FP16 engine. On Jetson Orin Nano this gives approximately 15–25 ms inference per tile vs 3 ms on T4, yielding ~8–12 FPS with SAHI at 4 tiles per frame.

**Step 2 — Reduce tile count for real-time**
Replace SAHI with a fixed 2×2 grid (4 tiles of 480×270 each) instead of dynamic slicing. This gives 4 inference calls per frame instead of 9–12, roughly tripling FPS to ~20–30 FPS.

**Step 3 — INT8 calibration**
With a calibration dataset of 100 VisDrone frames, INT8 quantization reduces model size to ~1.6 MB and inference time to ~8 ms per tile, enabling real-time processing at 25 FPS.

**Step 4 — ByteTrack CPU offload**
ByteTrack's Kalman update and IoU matching run on CPU and take ~2 ms — negligible. No GPU needed for tracking.

**Memory footprint on Jetson:**
- TensorRT FP16 engine: ~3 MB
- Runtime peak GPU memory: ~180 MB (tiles in batch)
- Total system RAM: ~400 MB including Python overhead

This fits comfortably within Jetson Orin Nano's 8 GB unified memory.

### Engineering trade-offs

| Trade-off | Choice made | Reason |
|---|---|---|
| Model size vs accuracy | YOLOv8n (6.3 MB) over YOLOv8s (22 MB) | Hard constraint: <300 MB; nano acceptable mAP for drone altitude |
| Recall vs precision | conf=0.28 daytime, 0.38 night | Night: glowing signs produce low-conf FPs; daytime: prefer catching all people |
| Speed vs coverage | SAHI 512px tiles | Full-res single pass misses small persons; SAHI recovers them at 3–4× FPS cost |
| Track stability vs ID uniqueness | match_thresh=0.60, buffer=60 | Drone motion makes strict IoU matching create many new IDs; relaxing improves continuity |
| Complexity vs weight | ByteTrack over DeepSORT | DeepSORT needs re-ID model (+50 MB); ByteTrack is weight-free with comparable performance |

### Known limitations
- Illuminated billboard characters in night sequences occasionally pass the detector's confidence threshold and geometry filters because backlit rectangular characters resemble human silhouettes. Mitigation: raising night confidence threshold to 0.38 reduces but does not eliminate these. Full elimination requires hard-negative mining and retraining with nighttime signage examples.
- FPS of 1.2–4.0 on T4 is below real-time due to SAHI overhead. On deployment hardware with TensorRT + fixed tiling, real-time is achievable (see edge section above).
- ID switching during close person interactions is a fundamental limitation of IoU-based tracking without appearance features.
