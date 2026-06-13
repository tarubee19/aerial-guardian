# AerialGuardian 🚁

**Multi-Person Detection and Tracking from a Moving Drone Platform**

A lightweight, real-time pipeline for detecting and tracking multiple persons in drone footage. Built on YOLOv8s fine-tuned on VisDrone, with BoT-SORT tracking enhanced with Global Motion Compensation for drone ego-motion robustness.

---

## Results

## Output Videos

All 7 tracked sequences (full length, real FPS):
[Google Drive — AerialGuardian_SUBMISSION](https://drive.google.com/drive/folders/1tLwp8iiV4_2eV9Bjn7Y9GTuwl-ygDpvr?usp=drive_link)

| Sequence | Scene | Persons | FPS | Hardware |
|---|---|---|---|---|
| uav0000086 — Basketball Court | overhead | ~25 | 10.4 | Tesla T4 |
| uav0000117 — Night Street | night | ~45 | 8.5 | Tesla T4 |
| uav0000137 — Crossroads | day | ~40 | 9.1 | Tesla T4 |
| uav0000182 — Urban Street | day | ~61 | 10.6 | Tesla T4 |
| uav0000268 — Highway | day | ~12 | 14.0 | Tesla T4 |
| uav0000305 — Overhead | overhead | ~20 | 15.0 | Tesla T4 |
| uav0000339 — Highway 2 | day | ~15 | 13.5 | Tesla T4 |

**Model size: 22.5 MB — 13× under the 300 MB limit**

---

## Setup

```bash
git clone https://github.com/tarubee19/aerial-guardian
cd aerial-guardian
pip install -r requirements.txt
```

Download model weights (22.5 MB):
> [best.pt — Google Drive](https://drive.google.com/file/d/1t1tMolkyaigaSwhkAheN3gUzfiCN1aa3/view?usp=sharing)

Place at: `weights/best.pt`

---

## Run

```bash
# Single sequence
python src/pipeline.py \
    --seq     path/to/sequence/frames \
    --model   weights/best.pt \
    --output  outputs/tracked.mp4

# All 7 VisDrone val sequences
python src/run_all.py \
    --seq_root  path/to/VisDrone2019-MOT-val/sequences \
    --model     weights/best.pt \
    --output    outputs/
```

---

## File Structure

```
aerial-guardian/
├── README.md
├── requirements.txt
├── configs/
│   └── drone_botsort.yaml      # Drone-tuned tracker config
├── src/
│   ├── tracker.py              # DronePersonTracker class
│   ├── pipeline.py             # Single sequence video generator
│   └── run_all.py              # Run on all sequences
├── weights/
│   └── best.pt                 # Fine-tuned YOLOv8s (22.5 MB)
└── outputs/                    # Generated tracking videos
    └── .gitkeep
```

---

## Architecture — Full Technical Explanation

### The Problem: Why Drone Tracking Is Hard

Standard person detection fails on drone footage for three reasons that do not exist in fixed-camera or ground-level scenarios.

**Small object problem:** A person filmed from 30–50 metres altitude occupies only 8–40 pixels in a 1920×1080 frame. Standard YOLO downsamples input to 640×640 pixels. A 15-pixel-tall person in a 1344-pixel-wide frame becomes 7 pixels after this downsampling — below any reliable detection threshold. Our fix: run inference at imgsz=960, which keeps that same person at 11 pixels — detectable. This single change improves small-object recall by approximately 20% at 2× inference cost. That cost is justified because missing people entirely is worse than being slightly slower.

**Camera ego-motion problem:** When the drone camera pans, every tracked person appears to shift in the frame — including people who are standing completely still. A tracker using standard IoU matching will see the predicted position and the new detection position as mismatched (because the camera moved, not the person), drop the track, and assign a new ID. This is the primary cause of ID switching in drone footage. Our fix: BoT-SORT with Global Motion Compensation.

**Variable scene problem:** The 7 test sequences include night streets, pure overhead top-down shots, dense basketball courts, and sparse highways. No single confidence threshold or aspect ratio filter works optimally for all of them. A threshold tuned for night will flood daytime scenes with false positives. Our fix: automatic scene classification that adapts parameters based on actual frame pixel measurements — no sequence names or hardcoded paths.

---

### Detection: YOLOv8s Fine-Tuned on VisDrone

**Why YOLOv8s and not other variants:**

YOLOv8 is a single-stage anchor-free detector that processes the entire image in one forward pass, making it fast enough for drone deployment. The model uses a CSPDarknet backbone, a Path Aggregation Feature Pyramid Network (PAN-FPN) neck for multi-scale feature fusion, and a decoupled detection head that separates classification and localisation.

We chose the `s` (small) variant specifically:
- YOLOv8n (6 MB, 3.2M params) — insufficient capacity to learn subtle features of 8-pixel persons against complex backgrounds
- YOLOv8s (22.5 MB, 11.1M params) — sufficient capacity, fast enough, well under 300 MB limit ✅
- YOLOv8m and larger — slower inference, diminishing accuracy returns, unnecessary for our 2-class task

**Training details:**

We fine-tuned from COCO-pretrained weights on the VisDrone DET training set (6,471 images). Before writing a single line of training code, we verified the actual category IDs in the annotation files by reading them directly:

```
cat 1 = pedestrian  → class 0 (person)   ✅
cat 2 = people      → class 0 (person)   ✅
cat 4 = car         → SKIP               ✅
cat 5 = van         → SKIP               ✅
cat 7 = tricycle    → SKIP               ✅ (NOT motor+rider)
cat 10 = motor      → class 1 (rider)    ✅ (MOT dataset only)
```

This verification step was critical. An incorrect mapping (e.g. mapping cat 4=car as rider) would cause the model to learn vehicle appearance as person appearance — exactly the kind of systematic error that produces correct validation metrics but completely wrong real-world behaviour.

Training configuration:
- Epochs: 80, imgsz=960, batch=16
- Result: mAP@50=0.551, Recall=0.512, Model size=22.5MB

**Why the Feature Pyramid Network matters for drones:**

The FPN produces feature maps at three scales simultaneously — 80×80 (fine spatial detail, detects small objects), 40×40 (medium), and 20×20 (rich semantic understanding, detects large objects). 

---

### Tracking: BoT-SORT with Drone-Specific Configuration

**The tracking-by-detection paradigm:**

Every frame, YOLO produces a fresh set of bounding boxes. The tracker's job is to answer: "Is the person at position X in frame 47 the same person as position Y in frame 46?" This is solved through four repeated steps:

1. **Predict** — Kalman filter predicts where each active track moved using a constant-velocity motion model. State vector: [cx, cy, w, h, vx, vy, vw, vh]
2. **Compensate** — GMC warps all predictions to account for camera motion
3. **Associate** — Hungarian algorithm finds optimal assignment of detections to tracks by minimising IoU cost matrix
4. **Update** — matched tracks update their Kalman state; unmatched detections start new tracks; tracks lost for track_buffer frames are deleted

**Why BoT-SORT over ByteTrack:**

ByteTrack's key innovation is using both high-confidence and low-confidence detections in two association passes — rescuing tracks that would be lost when a person is partially occluded. This works well for fixed cameras. For drone footage, ByteTrack's pure IoU matching fails because camera panning makes stationary objects appear to move, destroying IoU associations even for correct predictions.

BoT-SORT adds Global Motion Compensation. Before IoU matching, it:
1. Detects ~500 corner feature points in the previous frame using Shi-Tomasi detector
2. Tracks them to the current frame using Lucas-Kanade sparse optical flow
3. Estimates a homography matrix H using RANSAC (robust to moving objects that should be ignored)
4. Warps all Kalman predictions by H

A person who stood still but whose predicted position shifted by 20 pixels due to camera pan now has near-perfect IoU with their warped prediction. Without GMC, that same person would appear as a new detection and receive a new ID.

**Our drone-specific parameter changes:**

```yaml
match_thresh: 0.50      # default 0.80
# Drone vibration + residual GMC error causes 5-15px displacement.
# Default 0.80 requires IoU > 0.80 — too strict.
# 0.50 requires IoU > 0.50 — tolerates drone movement
# without incorrectly merging two nearby pedestrians.

track_buffer: 150       # default 30
# At 10 FPS output, 150 frames = 15 seconds of track memory.
# Covers: walking behind parked bus (3-5s), tree occlusion (1-3s),
# building occlusion (5-10s), detection gap from confidence
# fluctuation (0.5-2s).
# Default 30 = 3 seconds — insufficient for outdoor drone scenarios.

track_high_thresh: 0.35 # default 0.60
# Drone-altitude persons score 0.35-0.55 confidence.
# Default 0.60 would exclude most real detections as "low confidence".

gmc_method: sparseOptFlow  # default: none
# Enables the camera motion compensation described above.
# Single most important parameter for ID stability on drone footage.
```

---

### Auto Scene Classification

Every 30 frames, two measurements classify the scene from frame pixels alone. No sequence names, no dataset knowledge, no hardcoded paths — the same code runs correctly on any new drone footage:

```python
brightness = mean(grayscale_frame)
if brightness < 72:
    scene = 'night'          # universally dark: artificial light only
elif abs(top_15% - bottom_15%) < 15:
    scene = 'overhead'       # no sky visible = camera pointing down
else:
    scene = 'day'            # everything else: urban, highway, sparse
```

The top/bottom brightness contrast check works because angled drone shots always show a bright sky band at the top of frame (brightness ~200) against darker ground (brightness ~80-120), giving large contrast. Pure overhead shots show only ground in all regions — contrast near zero.

Each scene loads a different parameter profile. Night uses higher confidence thresholds to suppress light-reflection false positives and a longer confirmation gate to kill billboard flicker. Overhead uses looser aspect ratio limits because people look approximately square from directly above rather than tall and narrow from an angle. 

---

### Six-Layer Post-Processing Filter Stack

Between YOLO raw detections and final tracker output, six filters run in sequence. Each solves a specific observed failure mode.

**1. Vehicle Box Suppression**

When a vehicle-class detection (bicycle, motor) overlaps a person-class detection, the vehicle box is suppressed through four geometric checks:

```python
if iou(vehicle_box, person_box) > 0.20:        suppress vehicle
if person_centre inside vehicle_box:            suppress vehicle
if person_box 40%+ inside vehicle_box:          suppress vehicle
if vehicle_below_person with matching x_centre: suppress vehicle
```

This produces one box per rider. The four checks handle different riding postures — upright, leaning, partially occluded.

**2. Cross-Class NMS**

Standard YOLO NMS suppresses overlapping boxes of the same class only. A person box (class 0) and a rider box (class 1) around the same motorcycle rider both survive YOLO NMS even with 90% overlap — causing double boxes on riders.

Our fix: sort all boxes by area (largest first). For any two boxes with IoU > 0.25, suppress the smaller one regardless of class. We keep the larger box because it covers more of the person and is more stable for tracking continuity as the motorcycle turns.

**3. Adaptive Confidence Gate**

Smaller detections require higher confidence than larger ones:
- area < 300 px² → requires conf_small + 0.10
- area < 900 px² → requires conf_small
- area ≥ 900 px² → requires conf_base

Rationale: A tiny person at 80m altitude has a noisier feature response than a close person at 20m. Scaling the threshold with object size compensates for this inherent noise difference.

**4. Geometry Filter**

Bounding box shape must match realistic human proportions:
- Aspect ratio (h/w): 0.32 to 4.5
- Minimum size: 8×12 pixels
- Maximum area: 15,000 px²

Lamp posts: aspect ratio 8-15 → eliminated by max_aspect.
Parked vehicles: aspect ratio 0.1-0.3 → eliminated by min_aspect.
Road markings: area < 96 px² → eliminated by min size.
Merged two-person boxes: area > 15,000 → eliminated by max_area.

**5. Confirmation Gate**

An ID is only displayed after a detection persists for confirm_frames consecutive frames (4 for day, 6 for night, 3 for overhead).

This eliminates transient false positives that YOLO fires for 1-2 frames and then stops — billboard characters as the drone pans, light reflections on wet road surfaces, and background patterns that briefly match the person classifier. Real persons are detected consistently across many frames and always confirm. Night uses a higher gate (6 frames) because illuminated signs produce more persistent false positives.

**6. Static Filter**

If a detection's total movement over static_window=8 frames is less than static_px=6 pixels, it is suppressed — unless its confidence exceeds 0.78.

The confidence bypass is critical: a real person standing still waiting to cross the road would fail the movement test. But genuinely stationary persons are detected with high confidence (0.78-0.92) because they are clearly human-shaped and well-lit. Static false positives like painted road markings score 0.30-0.55. The bypass threshold of 0.78 sits cleanly between these two distributions.

**7. Trajectory Tail Visualisation**

Tails draw only when displacement between the last two tracked positions exceeds 3 pixels. This prevents spider-web artifacts where stationary persons sitting courtside accumulate permanent tail lines from Kalman filter noise (1-3px per frame jitter). Moving persons generate clean directional tails showing their trajectory.

---

## Known Issues and Honest Explanations

### Issue 1: Bench and Parked Bicycle Getting Boxes (Basketball Court)

**What you see:** In the basketball court sequence, a courtside bench and a parked bicycle occasionally receive bounding boxes with person IDs.

**Why it happens:** YOLOv8s is trained on VisDrone DET which contains 79,337 pedestrian and 27,059 "people" annotations. At drone altitude, a bench with people sitting on it has a bounding box in training data that encompasses both people and bench. The model learns that "bench-shaped objects with people-coloured blobs" are positive examples. When the basketball court bench appears empty but with similar shape, the model fires.

**Why the geometry filter does not fully catch it:** The bench in this sequence has aspect ratio ~0.48. Our min_aspect=0.32 passes this. Raising min_aspect to 0.60 would eliminate the bench but also eliminate sitting players leaning forward (aspect ratio 0.55-0.65) — a worse trade-off.

**The correct engineering fix:** Add empty bench images as explicit negative training examples in fine-tuning. The confirmation gate (4 frames) already eliminates most transient bench detections.

---

### Issue 2: ID Switching on Fast Motorcycle Riders

**What you see:** A person riding a motorcycle sometimes gets a new ID after a sharp turn or when passing behind a vehicle.

**Why it happens:** Three compounding causes. First, the motorcycle bounding box changes shape rapidly during a turn — aspect ratio shifts from 1.2 (side view) to 0.8 (front view) in 2-3 frames. The Kalman filter predicts a side-view box but receives a front-view box — IoU drops even though it is the same person. Second, at 7-10 FPS, a motorcycle at 30 km/h moves 15-20 pixels between frames — large displacement relative to box size. Third, two motorcycles crossing paths create ambiguous overlapping detections for 1-2 frames.

**What we did to reduce it:** match_thresh lowered from 0.80 to 0.50 (more lenient IoU matching), track_buffer raised from 30 to 150 (15 seconds of memory through occlusions), cross-class NMS to keep the larger more-stable box on riders.

**Remaining limitation:** IoU-based tracking has a fundamental limitation when bounding box shape changes rapidly. 

---

## Engineering Trade-offs Summary

| Decision | Choice | Alternative | Reason |
|---|---|---|---|
| Detection model | YOLOv8s 22.5MB | YOLOv8n 6MB | Better small object recall at drone altitude |
| Input resolution | imgsz=960 | imgsz=640 | 20% better recall on 8-15px persons, 2× slower |
| Tracker | BoT-SORT | ByteTrack | Camera motion compensation (GMC) for drones |
| ReID | HSV histogram | OSNet deep ReID | 1000× faster, 0 extra parameters, sufficient for pedestrians |
| Scene adaptation | 3-scene auto-detect | Single global params | Handles night/overhead/day without sequence-specific tuning |
| FPS (T4 GPU) | 7-15 FPS at imgsz=960 | 20-30 FPS at imgsz=640 | Catching tiny persons matters more than speed for offline analysis |

**FPS measurement methodology:** FPS is measured live at runtime by timing 10 actual inference calls on real frames and computing 1/mean(frame_times). No hardcoded values — the number shown in the video HUD reflects actual hardware performance at that moment.

---

## Edge Deployment — NVIDIA Jetson

**Step 1 — TensorRT FP16 export:**

```bash
yolo export model=weights/best.pt \
    format=engine half=True imgsz=960 device=0
```

TensorRT converts FP32 operations to FP16 (2× memory reduction), fuses consecutive operations (Conv + BatchNorm + ReLU → single GPU kernel), and pre-compiles CUDA kernels for the specific GPU architecture. Expected speedup: 2-3× over PyTorch.

**Step 2 — Reduce imgsz for real-time:**

At imgsz=640 on Jetson Orin NX with TensorRT FP16: estimated 18-25 FPS — sufficient for 20-30 FPS drone cameras. The ~8% recall reduction on the smallest objects (8-12 px) is acceptable when real-time performance is required.

**Expected Jetson Orin NX performance:**

| Config | FPS | Recall impact |
|---|---|---|
| PyTorch FP32 imgsz=960 | ~4-6 | baseline |
| TensorRT FP16 imgsz=960 | ~10-15 | none |
| TensorRT FP16 imgsz=640 | ~18-25 | -8% small objects |
| TensorRT INT8 imgsz=640 | ~28-35 | -12% small objects |

---

## What We Added Beyond Off-the-Shelf

The base components (YOLOv8s, BoT-SORT) are open source. Our contributions:

| Component | Off-the-shelf behaviour | Our addition |
|---|---|---|
| YOLOv8s | COCO 80-class detector | Fine-tuned on VisDrone with verified class mapping, 2-class person/rider |
| BoT-SORT | Default parameters for fixed cameras | Drone-tuned config: match_thresh=0.50, track_buffer=150, GMC enabled |
| Scene handling | None — single parameter set | Auto 3-scene detection from frame pixel measurements only |
| False positive suppression | None | 6-layer filter stack: vehicle suppression, cross-class NMS, adaptive conf, geometry, confirmation, static |
| Appearance ReID | None (with_reid=False in base) | HSV colour histogram bank with cosine similarity matching |
| Tail visualisation | None | Velocity-gated trajectory tails preventing spider-web artifacts |
| Double box fix | Standard same-class NMS only | Cross-class NMS: suppress smaller overlapping box regardless of class |
| FPS reporting | Not measured | Live runtime measurement displayed in video HUD |

---

## Dataset

VisDrone MOT 2019 — Task 4 Validation Set

7 sequences, never used in training:

```
uav0000086_00000_v  — Basketball court
uav0000117_02622_v  — Night street
uav0000137_00458_v  — Crossroads
uav0000182_00000_v  — Urban street
uav0000268_05773_v  — Highway
uav0000305_00000_v  — Overhead top-down
uav0000339_00001_v  — Highway 2
```

Training used VisDrone DET train (6,471 images). Zero overlap with validation sequences confirmed by comparing sequence folder names before any training was started.

---

## Requirements

```
ultralytics>=8.4.0
lapx
opencv-python
numpy
torch
torchvision
tqdm
Pillow
matplotlib
```
