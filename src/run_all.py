"""
AerialGuardian — run_all.py
============================
Run tracking pipeline on all 7 VisDrone MOT val sequences.

Usage:
    python src/run_all.py \
        --seq_root  path/to/VisDrone2019-MOT-val/sequences \
        --model     weights/best.pt \
        --output    outputs/
"""

import os
import argparse
from pipeline import run


SEQUENCES = [
    'uav0000086_00000_v',
    'uav0000117_02622_v',
    'uav0000137_00458_v',
    'uav0000182_00000_v',
    'uav0000268_05773_v',
    'uav0000305_00000_v',
    'uav0000339_00001_v',
]

LABELS = {
    'uav0000086_00000_v': 'basketball_court',
    'uav0000117_02622_v': 'night_street',
    'uav0000137_00458_v': 'crossroads',
    'uav0000182_00000_v': 'urban_street',
    'uav0000268_05773_v': 'highway',
    'uav0000305_00000_v': 'overhead',
    'uav0000339_00001_v': 'highway2',
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AerialGuardian — all sequences")
    parser.add_argument(
        '--seq_root',
        required=True,
        help='Path to VisDrone2019-MOT-val/sequences')
    parser.add_argument(
        '--model',
        required=True,
        help='Path to best.pt weights')
    parser.add_argument(
        '--output',
        required=True,
        help='Output folder for MP4 files')
    parser.add_argument(
        '--tracker',
        default='configs/drone_botsort.yaml',
        help='BoT-SORT config yaml')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    fps_log = {}

    print("AerialGuardian — Processing all 7 sequences")
    print("=" * 50)

    for seq in SEQUENCES:
        seq_dir = os.path.join(args.seq_root, seq)
        if not os.path.exists(seq_dir):
            print(f"⚠️  Skipping {seq} — not found")
            continue
        label      = LABELS[seq]
        output_mp4 = os.path.join(
            args.output, f"tracked_{label}.mp4")
        fps = run(seq_dir, args.model,
                  output_mp4, args.tracker)
        fps_log[label] = fps

    print("\n" + "=" * 50)
    print("COMPLETE — FPS Summary")
    print("=" * 50)
    for label, fps in fps_log.items():
        if fps:
            print(f"  {label:25s}: {fps:.1f} FPS")
    print(f"\nVideos saved to: {args.output}")
