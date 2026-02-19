#!/usr/bin/env python3
"""
Run Bundle Adjustment for a specific session.

Usage:
    python run_ba_session.py <session_id>

Example:
    python run_ba_session.py 8928f98a-b411-497e-aa4b-aa752434686d
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run bundle adjustment for a specific session')
    parser.add_argument('session_id', type=str, help='The session UUID (e.g., 3bcb81b4-d9ca-4fc9-a1cd-353a966239ca)')
    args = parser.parse_args()

    session_id = args.session_id
    
    # Define paths
    prediction_dir = "/teamspace/studios/this_studio/outputs/ibl-mouse/test_200_MVT_3d_aug_patch_masking/multiview_transformer_200_0-2/ensemble_median/videos_new_add"
    ba_script = "/teamspace/studios/this_studio/bundle_adjust_predictions.py"
    initial_calib = "/teamspace/studios/this_studio/calibrations/calibration.toml"
    output_calib = f"/teamspace/studios/this_studio/calibrations/_iblrig.downsampled.{session_id}.toml"

    # Define CSV files
    left_csv = f"{prediction_dir}/_iblrig_leftCamera.downsampled.{session_id}.csv"
    right_csv = f"{prediction_dir}/_iblrig_rightCamera.downsampled.{session_id}.csv"

    # Check if files exist
    if not Path(left_csv).exists():
        print(f"Error: Left camera CSV not found: {left_csv}")
        sys.exit(1)
    if not Path(right_csv).exists():
        print(f"Error: Right camera CSV not found: {right_csv}")
        sys.exit(1)
    if not Path(initial_calib).exists():
        print(f"Error: Initial calibration file not found: {initial_calib}")
        sys.exit(1)

    # Construct the command
    cmd = [
        "python", ba_script,
        "--csv_files", left_csv, right_csv,
        "--calibration_file", initial_calib,
        "--output_file", output_calib,
        "--likelihood_threshold", "0.95", # 0.95 in general
        "--variance_percentile", "5",
        "--loss", "linear",
        "--only_extrinsics"
    ]

    print(f"\nRunning Bundle Adjustment for session: {session_id}")
    print(f"Command: {' '.join(cmd)}\n")

    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Bundle adjustment failed for session {session_id}")
        sys.exit(1)

if __name__ == "__main__":
    main()
