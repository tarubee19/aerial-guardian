"""
AerialGuardian — DronePersonTracker
====================================
Single-model, single-tracker pipeline for drone-based
multi-person detection and tracking.

Automatically classifies scene type (night/overhead/day)
from frame pixel properties and adapts parameters.
No per-sequence hardcoding — works on any drone footage.

Author: AerialGuardian
Dataset: VisDrone MOT 2019 Task 4 Validation Set
Model: YOLOv8s fine-tuned on VisDrone DET
Tracker: BoT-SORT with Global Motion Compensation
"""

import cv2
import numpy as np
from collections import defaultdict, deque, Counter
from ultralytics import YOLO


class DronePersonTracker:
    """
    Complete drone person tracking pipeline.

    Scene profiles (auto-detected, not hardcoded):
        night    — brightness < 72 (artificial lighting only)
        overhead — top/bottom contrast < 15 (camera pointing down)
        day      — everything else (urban, sparse, highway)

    Five post-processing filters:
        1. Adaptive confidence gate (size-aware threshold)
        2. Geometry filter (aspect ratio + area)
        3. Confirmation gate (N consecutive frames required)
        4. Static filter (movement threshold with conf bypass)
        5. Vehicle suppression (4-check geometric algorithm)
    """

    SCENE_PROFILES = {
        'night': dict(
            conf_base=0.40,
            conf_small=0.52,
            min_w=12,
            min_h=18,
            max_aspect=3.8,
            min_aspect=0.35,
            max_area=9000,
            confirm_frames=6,
            tail_len=15,
            static_px=5,
            static_window=10,
            max_active=80,
            reid_thresh=0.80,
            reid_dist=100,
        ),
        'overhead': dict(
            conf_base=0.28,
            conf_small=0.40,
            min_w=7,
            min_h=7,
            max_aspect=3.5,
            min_aspect=0.50,
            max_area=12000,
            confirm_frames=3,
            tail_len=12,
            static_px=4,
            static_window=8,
            max_active=150,
            reid_thresh=0.74,
            reid_dist=80,
        ),
        'day': dict(
            conf_base=0.30,
            conf_small=0.42,
            min_w=8,
            min_h=12,
            max_aspect=4.5,
            min_aspect=0.32,
            max_area=15000,
            confirm_frames=4,
            tail_len=20,
            static_px=6,
            static_window=8,
            max_active=150,
            reid_thresh=0.76,
            reid_dist=130,
        ),
    }

    # VisDrone class IDs
    VEHICLE_CLASSES = {2, 3}   # bicycle, motor
    PERSON_CLASSES  = {0, 1}   # person, rider

    def __init__(self, model_path,
                 tracker_yaml='configs/drone_botsort.yaml'):
        self.model           = YOLO(model_path)
        self.tracker_yaml    = tracker_yaml
        self.scene_type      = 'day'
        self.p               = self.SCENE_PROFILES['day'].copy()
        self.track_history   = defaultdict(
            lambda: deque(maxlen=50))
        self.confirm_count   = defaultdict(int)
        self.recent_pos      = defaultdict(
            lambda: deque(maxlen=12))
        self.colour_cache    = {}
        self.appearance_bank = {}
        self.dead_tracks     = {}
        self.track_scores    = defaultdict(float)
        self.REID_TTL        = 90
        self.frame_idx       = 0
        self._scene_vote     = deque(maxlen=7)

    # ── Scene Detection ───────────────────────────────────

    def _auto_detect_scene(self, frame):
        """
        Classify scene using frame pixel properties only.
        No sequence names or dataset knowledge used.

        Measurements:
            brightness  = mean of grayscale frame
            contrast    = abs(top 15% mean - bottom 15% mean)
                          Large contrast = sky visible = angled shot
                          Small contrast = no sky = overhead shot
        """
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # Dark scene — artificial lighting only
        if brightness < 72:
            return 'night'

        # No brightness difference top to bottom
        # = camera pointing straight down, no sky visible
        h        = frame.shape[0]
        top_mean = np.mean(gray[:int(h * 0.15), :])
        bot_mean = np.mean(gray[int(h * 0.85):, :])
        if abs(float(top_mean) - float(bot_mean)) < 15:
            return 'overhead'

        return 'day'

    def _update_scene(self, frame):
        """Majority vote over last 7 classifications
        to prevent rapid scene type switching."""
        self._scene_vote.append(
            self._auto_detect_scene(frame))
        winner = Counter(
            self._scene_vote).most_common(1)[0][0]
        if winner != self.scene_type:
            self.scene_type = winner
            self.p = self.SCENE_PROFILES[winner].copy()
            self.track_history = defaultdict(
                lambda: deque(
                    maxlen=self.p['tail_len']))
            self.recent_pos = defaultdict(
                lambda: deque(
                    maxlen=self.p['static_window']))

    # ── Geometry ──────────────────────────────────────────

    def _box_iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        if inter == 0:
            return 0.0
        union = ((b1[2]-b1[0]) * (b1[3]-b1[1]) +
                 (b2[2]-b2[0]) * (b2[3]-b2[1]) -
                 inter + 1e-6)
        return inter / union

    def _geometry_ok(self, x1, y1, x2, y2):
        """
        Reject non-human shaped detections.
        Aspect ratio h/w:
            Lamp posts: 8-15  → killed by max_aspect=4.5
            Vehicles:   0.1-0.3 → killed by min_aspect=0.32
            Humans:     0.5-4.0 → pass
        """
        w    = x2 - x1
        h    = y2 - y1
        asp  = h / (w + 1e-6)
        area = w * h
        if w    < self.p['min_w']:        return False
        if h    < self.p['min_h']:        return False
        if asp  > self.p['max_aspect']:   return False
        if asp  < self.p['min_aspect']:   return False
        if area > self.p['max_area']:     return False
        return True

    def _adaptive_conf(self, x1, y1, x2, y2):
        """
        Smaller detections need higher confidence.
        Tiny far-away persons have noisier feature responses.
        """
        area = (x2 - x1) * (y2 - y1)
        if area < 300:
            return self.p['conf_small'] + 0.10
        if area < 900:
            return self.p['conf_small']
        return self.p['conf_base']

    # ── Filters ───────────────────────────────────────────

    def _suppress_vehicles(self, boxes, ids,
                           confs, classes):
        """
        Remove vehicle boxes that overlap person boxes.
        4 checks handle different rider postures:
        1. IoU overlap > 0.20
        2. Person centre inside vehicle box
        3. Person box 40%+ inside vehicle box
        4. Vehicle directly below person (rider posture)
        """
        if len(boxes) == 0:
            return boxes, ids, confs, classes
        keep = np.ones(len(boxes), dtype=bool)
        pi = np.where([c in self.PERSON_CLASSES
                       for c in classes])[0]
        vi = np.where([c in self.VEHICLE_CLASSES
                       for c in classes])[0]
        for v in vi:
            vbox  = boxes[v]
            v_cx  = (vbox[0] + vbox[2]) / 2
            v_w   = vbox[2] - vbox[0]
            for p in pi:
                pbox   = boxes[p]
                p_cx   = (pbox[0] + pbox[2]) / 2
                p_cy   = (pbox[1] + pbox[3]) / 2
                p_area = ((pbox[2]-pbox[0]) *
                          (pbox[3]-pbox[1]))
                ix1 = max(vbox[0], pbox[0])
                iy1 = max(vbox[1], pbox[1])
                ix2 = min(vbox[2], pbox[2])
                iy2 = min(vbox[3], pbox[3])
                inter = (max(0, ix2-ix1) *
                         max(0, iy2-iy1))
                if self._box_iou(vbox, pbox) > 0.20:
                    keep[v] = False; break
                if (vbox[0] < p_cx < vbox[2] and
                        vbox[1] < p_cy < vbox[3]):
                    keep[v] = False; break
                if (p_area > 0 and
                        inter / p_area > 0.40):
                    keep[v] = False; break
                if (abs(v_cx - p_cx) < (v_w * 0.6) and
                        vbox[1] > pbox[1] and
                        vbox[1] < p_cy < vbox[3] + 20):
                    keep[v] = False; break
        idx = np.where(keep)[0]
        return (boxes[idx], ids[idx],
                confs[idx], classes[idx])

    def _kill_merged_boxes(self, boxes, ids,
                           confs, classes):
        """
        Remove large boxes that contain 2+ smaller boxes.
        These are merged detections of multiple people.
        """
        if len(boxes) < 2:
            return boxes, ids, confs, classes
        keep = np.ones(len(boxes), dtype=bool)
        for i in range(len(boxes)):
            if not keep[i]:
                continue
            contained = 0
            for j in range(len(boxes)):
                if i == j or not keep[j]:
                    continue
                area_j = ((boxes[j][2]-boxes[j][0]) *
                           (boxes[j][3]-boxes[j][1]))
                ix1 = max(boxes[i][0], boxes[j][0])
                iy1 = max(boxes[i][1], boxes[j][1])
                ix2 = min(boxes[i][2], boxes[j][2])
                iy2 = min(boxes[i][3], boxes[j][3])
                inter = (max(0, ix2-ix1) *
                         max(0, iy2-iy1))
                if area_j > 0 and inter/area_j > 0.65:
                    contained += 1
            if contained >= 2:
                keep[i] = False
        idx = np.where(keep)[0]
        return (boxes[idx], ids[idx],
                confs[idx], classes[idx])

    def _is_static(self, tid, conf):
        """
        Suppress non-moving detections.
        High-confidence bypass (conf > 0.78) keeps
        genuinely stationary persons (waiting to cross etc.)
        """
        if conf > 0.78:
            return False
        pts = list(self.recent_pos[tid])
        if len(pts) < self.p['static_window']:
            return False
        total = sum(
            np.sqrt((pts[i][0]-pts[i-1][0])**2 +
                    (pts[i][1]-pts[i-1][1])**2)
            for i in range(1, len(pts)))
        return total < self.p['static_px']

    # ── Appearance ReID ───────────────────────────────────

    def _get_appearance(self, frame, x1, y1, x2, y2):
        """HSV colour histogram for appearance matching."""
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame.shape[1], int(x2))
        y2 = min(frame.shape[0], int(y2))
        crop = frame[y1:y2, x1:x2]
        if (crop.size == 0 or
                (x2-x1) < 8 or (y2-y1) < 8):
            return None
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _similarity(self, h1, h2):
        """Cosine similarity between two histograms."""
        if h1 is None or h2 is None:
            return 0.0
        n1 = np.linalg.norm(h1)
        n2 = np.linalg.norm(h2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(h1, h2) / (n1 * n2))

    # ── Visualisation ─────────────────────────────────────

    def _get_colour(self, tid):
        """Consistent colour per track ID."""
        if tid not in self.colour_cache:
            np.random.seed(int(tid) % 10000)
            h   = int(np.random.randint(0, 180))
            bgr = cv2.cvtColor(
                np.uint8([[[h, 210, 220]]]),
                cv2.COLOR_HSV2BGR)[0][0]
            self.colour_cache[tid] = (
                int(bgr[0]),
                int(bgr[1]),
                int(bgr[2]))
        return self.colour_cache[tid]

    def _draw_tail(self, frame, tid, colour, score):
        """
        Velocity-gated trajectory tail.
        Only draws when displacement > 3px to prevent
        spider-web artifacts for stationary persons.
        """
        pts = list(self.track_history[tid])
        if len(pts) < 2:
            return
        dx = pts[-1][0] - pts[-2][0]
        dy = pts[-1][1] - pts[-2][1]
        if np.sqrt(dx**2 + dy**2) < 3.0:
            return
        for i in range(1, len(pts)):
            alpha = (i/len(pts)) * min(1.0, score*1.2)
            thick = max(1, int(2 * alpha))
            c     = tuple(int(x*alpha) for x in colour)
            cv2.line(
                frame,
                (int(pts[i-1][0]), int(pts[i-1][1])),
                (int(pts[i][0]),   int(pts[i][1])),
                c, thick, cv2.LINE_AA)

    def _draw_hud(self, frame, count, fps=0):
        """On-frame HUD with frame info and hardware."""
        for txt, y, col in [
            (f"Frame {self.frame_idx} | "
             f"Persons: {count} | {self.scene_type}",
             32, (255, 255, 255)),
            (f"HW: Tesla T4 GPU | "
             f"Tracker: BoT-SORT | FPS: {fps:.1f}",
             62, (0, 220, 255))]:
            cv2.putText(frame, txt, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                col, 2, cv2.LINE_AA)

    # ── Main Processing Loop ──────────────────────────────

    def process_frame(self, frame, fps=0):
        """
        Process one frame through the full pipeline.

        Pipeline order:
        1. Scene detection (every 30 frames)
        2. YOLO detection + BoT-SORT tracking
        3. Vehicle suppression
        4. Merged box suppression
        5. Per-detection filters (conf, geometry,
           static, confirmation)
        6. Appearance ReID for re-emerging tracks
        7. Visualisation (boxes, IDs, tails, HUD)
        """
        self.frame_idx += 1

        # Update scene every 30 frames
        if self.frame_idx % 30 == 1:
            self._update_scene(frame)

        # Run YOLO + BoT-SORT
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_yaml,
            conf=max(0.18, self.p['conf_base'] - 0.12),
            iou=0.45,
            verbose=False,
            imgsz=960)

        out = frame.copy()

        if results[0].boxes.id is None:
            self._draw_hud(out, 0, fps)
            return out

        boxes   = results[0].boxes.xyxy.cpu().numpy()
        ids     = results[0].boxes.id.cpu().numpy(
                      ).astype(int)
        confs   = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy(
                      ).astype(int)

        # Filter 1: vehicle suppression
        boxes, ids, confs, classes = \
            self._suppress_vehicles(
                boxes, ids, confs, classes)

        # Filter 2: merged box suppression
        boxes, ids, confs, classes = \
            self._kill_merged_boxes(
                boxes, ids, confs, classes)

        # Expire dead tracks past TTL
        self.dead_tracks = {
            k: v for k, v in self.dead_tracks.items()
            if (self.frame_idx - v['frame_lost']
                <= self.REID_TTL)}

        # Move lost tracks to dead bank
        active_ids = set(ids.tolist())
        for tid in list(self.appearance_bank.keys()):
            if tid not in active_ids:
                lp = list(self.recent_pos[tid])
                self.dead_tracks[tid] = {
                    'appearance':
                        self.appearance_bank[tid],
                    'last_pos':
                        lp[-1] if lp else (0, 0),
                    'frame_lost': self.frame_idx}
                del self.appearance_bank[tid]

        # Cap active tracks at max_active
        if len(active_ids) > self.p['max_active']:
            allowed = set(sorted(
                active_ids,
                key=lambda t: self.track_scores.get(t, 0),
                reverse=True)[:self.p['max_active']])
        else:
            allowed = active_ids

        confirmed_count = 0

        for box, tid, conf, cls in zip(
                boxes, ids, confs, classes):

            if tid not in allowed:             continue
            if cls not in self.PERSON_CLASSES: continue

            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Filter 3: adaptive confidence gate
            if conf < self._adaptive_conf(
                    x1, y1, x2, y2):
                continue

            # Filter 4: geometry filter
            if not self._geometry_ok(x1, y1, x2, y2):
                continue

            self.recent_pos[tid].append((cx, cy))

            # Filter 5: static filter
            if self._is_static(tid, conf):
                continue

            # Update track score
            self.track_scores[tid] = min(
                1.0,
                self.track_scores.get(tid, 0) +
                conf * 0.35)
            score = self.track_scores[tid]

            # Confirmation gate
            self.confirm_count[tid] += 1
            confirmed = (self.confirm_count[tid] >=
                         self.p['confirm_frames'])
            if confirmed:
                confirmed_count += 1

            # ReID: match re-emerging persons
            app = self._get_appearance(
                frame, x1, y1, x2, y2)
            if app is not None:
                self.appearance_bank[tid] = app
                for dtid, info in list(
                        self.dead_tracks.items()):
                    dx = cx - info['last_pos'][0]
                    dy = cy - info['last_pos'][1]
                    if (np.sqrt(dx*dx + dy*dy) >
                            self.p['reid_dist']):
                        continue
                    if (self._similarity(
                            app,
                            info['appearance']) >
                            self.p['reid_thresh']):
                        del self.dead_tracks[dtid]
                        break

            self.track_history[tid].append((cx, cy))

            if score < 0.25:
                continue

            # Draw
            colour = self._get_colour(tid)
            self._draw_tail(out, tid, colour, score)

            thick = 2 if confirmed else 1
            cv2.rectangle(
                out,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                colour, thick)

            if confirmed:
                lbl = f"P{tid}"
                (lw, lh), _ = cv2.getTextSize(
                    lbl, cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, 1)
                cv2.rectangle(
                    out,
                    (int(x1), int(y1)-lh-5),
                    (int(x1)+lw+4, int(y1)),
                    colour, -1)
                cv2.putText(
                    out, lbl,
                    (int(x1)+2, int(y1)-3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 1, cv2.LINE_AA)

        self._draw_hud(out, confirmed_count, fps)
        return out
