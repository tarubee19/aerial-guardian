"""
AerialGuardian — pipeline.py
=============================
Generate tracked output video from a sequence of frames.
FPS is measured from actual runtime — not hardcoded.

Usage:
    python src/pipeline.py \
        --seq     path/to/frames \
        --model   weights/best.pt \
        --output  outputs/tracked.mp4
"""

import os
import cv2
import glob
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from tqdm import tqdm

from tracker import DronePersonTracker


def run(seq_dir, model_path, output_path,
        tracker_yaml='configs/drone_botsort.yaml'):
    """
    Process a sequence of JPG frames and write
    a tracked MP4 video at real processing FPS.

    FPS measurement: time 10 warmup frames,
    compute 1/mean(frame_times). Written to video
    so playback matches actual processing speed.

    Args:
        seq_dir      : folder containing JPG frames
        model_path   : path to best.pt weights
        output_path  : output MP4 file path
        tracker_yaml : BoT-SORT config file path
    """
    # Find frames
    imgs = sorted(glob.glob(f'{seq_dir}/*.jpg'))
    if not imgs:
        imgs = sorted(
            glob.glob(f'{seq_dir}/img1/*.jpg'))
    if not imgs:
        raise FileNotFoundError(
            f"No JPG frames found in {seq_dir}")

    print(f"\nSequence : {seq_dir}")
    print(f"Frames   : {len(imgs)}")
    print(f"Model    : {model_path}")

    # Output dimensions
    first       = cv2.imread(imgs[0])
    H, W        = first.shape[:2]
    scale       = 1280 / W if W > 1280 else 1.0
    out_w       = int(W * scale)
    out_h       = int(H * scale)

    # Measure real processing FPS on warmup frames
    print("Measuring real FPS on 10 warmup frames...")
    test_tracker = DronePersonTracker(
        model_path, tracker_yaml)
    warmup_times = []
    for i in range(min(10, len(imgs))):
        frame = cv2.imread(imgs[i])
        t0    = time.time()
        test_tracker.process_frame(frame)
        warmup_times.append(time.time() - t0)
    del test_tracker

    real_fps  = 1.0 / np.mean(warmup_times)
    write_fps = max(5.0, min(30.0, real_fps))
    print(f"Real FPS : {real_fps:.1f} → "
          f"writing at {write_fps:.1f} FPS")

    # Set up video writer
    os.makedirs(
        os.path.dirname(
            os.path.abspath(output_path)),
        exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        write_fps,
        (out_w, out_h))

    # Process all frames
    tracker     = DronePersonTracker(
        model_path, tracker_yaml)
    frame_times = deque(maxlen=30)

    for img_path in tqdm(imgs, desc="Tracking"):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        live_fps = (1.0 / np.mean(frame_times)
                    if frame_times else real_fps)
        t0     = time.time()
        result = tracker.process_frame(
            frame, fps=live_fps)
        frame_times.append(time.time() - t0)
        if scale != 1.0:
            result = cv2.resize(
                result, (out_w, out_h))
        writer.write(result)

    writer.release()
    avg = (1.0 / np.mean(frame_times)
           if frame_times else 0)

    print(f"\nDone")
    print(f"Output   : {output_path}")
    print(f"Avg FPS  : {avg:.1f}")
    print(f"Scene    : {tracker.scene_type}")
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AerialGuardian — single sequence")
    parser.add_argument(
        '--seq',
        required=True,
        help='Path to folder of JPG frames')
    parser.add_argument(
        '--model',
        required=True,
        help='Path to best.pt weights')
    parser.add_argument(
        '--output',
        required=True,
        help='Output MP4 path')
    parser.add_argument(
        '--tracker',
        default='configs/drone_botsort.yaml',
        help='BoT-SORT config yaml')
    args = parser.parse_args()
    run(args.seq, args.model,
        args.output, args.tracker)
