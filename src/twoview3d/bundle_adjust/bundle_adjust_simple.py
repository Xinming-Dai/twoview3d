#!/usr/bin/env python3
"""
Simple Bundle Adjustment Script - Matching lightning-pose-app Implementation

This script performs bundle adjustment exactly as implemented in the lightning-pose-app
repository, using aniposelib's CameraGroup for bundle adjustment.

Usage:
    python bundle_adjust_simple.py \
        --csv_files path/to/view1.csv path/to/view2.csv ... \
        --calibration_file path/to/calibration.toml \
        --output_file path/to/output_calibration.toml \
        --session_filter session_name \
        [--use_iterative]

Example:
    python bundle_adjust_simple.py \
        --csv_files data/CollectedData_leftCamera.csv data/CollectedData_rightCamera.csv \
        --calibration_file calibrations/session.toml \
        --output_file calibrations/session_ba.toml \
        --session_filter "downsampled.2bdf206a"
        
    # Use iterative bundle adjustment with increased iterations
    python bundle_adjust_simple.py \
        --csv_files data/CollectedData_leftCamera.csv data/CollectedData_rightCamera.csv \
        --calibration_file calibrations/session.toml \
        --output_file calibrations/session_ba.toml \
        --session_filter "downsampled.2bdf206a" \
        --use_iterative \
        --max_iterations 500 \
        --ftol 1e-10
    
    python bundle_adjust_simple.py \
        --csv_files data/CollectedData_leftCamera.csv data/CollectedData_rightCamera.csv \
        --calibration_file calibration.toml \
        --output_file calibrations/session_ba.toml \
        --session_filter "downsampled.2bdf206a"
    
    # this is for InD data
    python bundle_adjust_simple.py \
        --csv_files "/teamspace/studios/data/ibl-mouse/CollectedData_Jan7/Original Labeled Data from labeler (new sessions)/CollectedData_leftCamera.csv" \
                    "/teamspace/studios/data/ibl-mouse/CollectedData_Jan7/Original Labeled Data from labeler (new sessions)/CollectedData_rightCamera.csv" \
        --calibration_file /teamspace/studios/this_studio/calibrations/calibration.toml \
        --output_file /teamspace/studios/this_studio/calibrations/_iblrig.downsampled.239dd3c9-35f3-4462-95ee-91b822a22e6b.toml \
        --session_filter "239dd3c9-35f3-4462-95ee-91b822a22e6b" \
        --only_extrinsics

    # this is for OOD data
    python bundle_adjust_simple.py \
        --csv_files "/teamspace/studios/data/ibl-mouse/CollectedData_Jan7/Original Labeled Data from labeler (new sessions)/CollectedData_leftCamera_new.csv" \
                    "/teamspace/studios/data/ibl-mouse/CollectedData_Jan7/Original Labeled Data from labeler (new sessions)/CollectedData_rightCamera_new.csv" \
        --calibration_file /teamspace/studios/this_studio/calibrations/calibration.toml \
        --output_file /teamspace/studios/this_studio/calibrations/_iblrig.downsampled.4b00df29-3769-43be-bb40-128b1cba6d35.toml \
        --session_filter "4b00df29-3769-43be-bb40-128b1cba6d35" \
        --only_extrinsics
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from aniposelib.cameras import CameraGroup

# Add lightning-pose to path if needed
lightning_pose_path = Path(__file__).parent / "lightning-pose"
if lightning_pose_path.exists() and str(lightning_pose_path) not in sys.path:
    sys.path.insert(0, str(lightning_pose_path))

try:
    from lightning_pose.utils.io import fix_empty_first_row
except ImportError:
    print("Warning: Could not import fix_empty_first_row from lightning_pose")
    def fix_empty_first_row(df):
        return df


def auto_label_session_key(frame_path: str, views: list) -> str | None:
    """
    Extract session key from frame path by replacing view names with *.
    Mirrors the logic from lightning-pose-app.
    """
    parts = frame_path.split("/")
    if len(parts) < 3:
        return None
    
    session_view_name_with_dots = parts[-2]  # e.g. 05272019_fly1_0_R1C24_Cam-A_rot-ccw-0.06_sec

    def process_part(session_view_name):
        """Mirrors frame.model.ts get autolabelSessionKey()"""
        # Replace view with *, e.g. 05272019_fly1_0_R1C24_*_rot-ccw-0.06_sec
        sessionkey_from_frame = re.sub(
            rf"({'|'.join([re.escape(_v) for _v in views])})", "*", session_view_name
        )

        # View not in this token, so return identity.
        if '*' not in sessionkey_from_frame:
            return sessionkey_from_frame

        # Attempt to parse assuming - is the delimiter.
        parts_hyphenated = sessionkey_from_frame.split('-')
        if '*' in parts_hyphenated:
            return '-'.join(filter(lambda x: x != '*', parts_hyphenated))

        # Attempt to parse assuming _ is the delimiter.
        parts_underscored = sessionkey_from_frame.split('_')
        if '*' in parts_underscored:
            return '_'.join(filter(lambda x: x != '*', parts_underscored))

        # View present, but invalid delimiter: return None
        return None

    # Split on . and process each part.
    processed_parts = list(map(process_part, session_view_name_with_dots.split('.')))
    # If some part had * but without correct delimiters around it, return null.
    if None in processed_parts:
        return None
    # Filter empty tokens after processPart (* got removed) and join by .
    return '.'.join(filter(lambda p: bool(p), processed_parts))


def is_of_current_session(img_path: str, session_key: str, views: list):
    """Check if image path belongs to current session."""
    return auto_label_session_key(img_path, views) == session_key


def load_multiview_data(csv_files, view_names, session_filter=None, keypoint_filter=None):
    """
    Load multiview data exactly as in lightning-pose-app.
    
    Args:
        csv_files: List of CSV file paths
        view_names: List of view/camera names
        session_filter: Optional session ID to filter frames
        keypoint_filter: Optional list of keypoint names to use (e.g., ['nose', 'paw1'])
                         If None, uses all keypoints
    
    Returns:
        p2ds: np.ndarray of shape (n_views, n_frames, n_keypoints, 2)
        views: list of view names
        keypoint_names: list of keypoint names used
    """
    print(f"\nLoading multiview data from {len(csv_files)} CSV files...")
    if session_filter:
        print(f"  Filtering for session: {session_filter}")
    
    # Read DFs
    dfs_by_view = {}
    for view, csv_file in zip(view_names, csv_files):
        print(f"  Loading {view}: {csv_file}")
        df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
        df = fix_empty_first_row(df)
        dfs_by_view[view] = df

    # Check that DFs are aligned
    index_values = dfs_by_view[view_names[0]].index.values
    firstview_framekeys = list(map(lambda s: s.replace(view_names[0], ""), index_values))
    for view in view_names:
        thisview_framekeys = list(
            map(lambda s: s.replace(view, ""), dfs_by_view[view].index.values)
        )
        if not firstview_framekeys == thisview_framekeys:
            print(f"Skipping {csv_files[view_names.index(view)]} because of misaligned indices")
            continue

    # Filter to frames of current session
    if session_filter:
        print(f"  Filtering for session: {session_filter}")
        for view in view_names:
            df = dfs_by_view[view]
            # Simple session filtering: check if the session ID is in the image path
            session_mask = df.index.str.contains(session_filter, na=False)
            dfs_by_view[view] = df[session_mask]
            print(f"    {view}: {len(dfs_by_view[view])} frames after filtering")
            if len(dfs_by_view[view]) == 0:
                raise RuntimeError(
                    f"Insufficient frames found after filtering for session {session_filter}. "
                    "Possible error in session extraction logic."
                )

    # Get keypoint names from first view's columns (BEFORE filtering NaNs)
    # Column structure: (scorer, bodypart_name, x/y)
    # Example: ('scorer', 'nose', 'x') -> keypoint is 'nose' at index 1
    first_view = view_names[0]
    first_df = dfs_by_view[first_view]
    all_keypoint_names = []
    for col in first_df.columns:
        if col[2] == "x":  # Only count x columns (y will follow)
            keypoint_name = col[1]  # Second level (index 1) is bodypart/keypoint name
            if keypoint_name not in all_keypoint_names:
                all_keypoint_names.append(keypoint_name)
    
    # Filter keypoints if requested (BEFORE filtering NaNs)
    if keypoint_filter is not None:
        # Convert to lowercase for case-insensitive matching
        keypoint_filter_lower = [k.lower() for k in keypoint_filter]
        filtered_keypoint_names = []
        for kp in all_keypoint_names:
            if kp.lower() in keypoint_filter_lower:
                filtered_keypoint_names.append(kp)
        
        if len(filtered_keypoint_names) == 0:
            raise RuntimeError(
                f"No matching keypoints found! Requested: {keypoint_filter}, "
                f"Available: {all_keypoint_names}"
            )
        
        print(f"\nKeypoint filtering:")
        print(f"  Available keypoints: {all_keypoint_names}")
        print(f"  Requested keypoints: {keypoint_filter}")
        print(f"  Using keypoints: {filtered_keypoint_names}")
        keypoint_names = filtered_keypoint_names
    else:
        keypoint_names = all_keypoint_names
        print(f"\nUsing all keypoints: {keypoint_names}")

    # Remove rows with NaN coordinates ONLY for the selected keypoints
    # This way, frames with NaN in other keypoints but valid data in selected keypoints are kept
    nan_indices = set()
    for view in view_names:
        df = dfs_by_view[view]
        # Only check columns for selected keypoints
        picked_columns = [c for c in df.columns if c[2] in ("x", "y") and c[1] in keypoint_names]
        if len(picked_columns) == 0:
            raise RuntimeError(
                f"No columns found for keypoints {keypoint_names} in view {view}"
            )
        # Find rows with NaN in any of the selected keypoint columns
        nan_rows = df.loc[:, picked_columns].dropna().index.symmetric_difference(df.index)
        nan_indices.update(df.index.get_indexer(nan_rows))

    # Drop those indices from all views  
    if len(nan_indices) > 0:
        print(f"\nDropping {len(nan_indices)} frames with NaN in selected keypoints ({keypoint_names})")
        for view in view_names:
            dfs_by_view[view] = dfs_by_view[view].drop(dfs_by_view[view].index[list(nan_indices)])
            print(f"  {view}: {len(dfs_by_view[view])} frames remaining")
            if len(dfs_by_view[view]) == 0:
                raise RuntimeError(
                    f"Insufficient frames found after dropping NaN rows for session {session_filter}. "
                    f"All frames had NaN values for keypoints {keypoint_names}."
                )
    else:
        print(f"\nAll frames have valid data for selected keypoints ({keypoint_names})")
    
    # Normalize columns: x, y alternating, filtered by keypoint names
    for view in view_names:
        df = dfs_by_view[view]
        # Filter columns to only include selected keypoints
        # Column structure: (scorer, bodypart_name, x/y)
        picked_columns = []
        for col in df.columns:
            if col[2] in ("x", "y") and col[1] in keypoint_names:  # col[1] is the keypoint name
                picked_columns.append(col)
        
        assert len(picked_columns) % 2 == 0, f"Expected even number of columns, got {len(picked_columns)}"
        assert len(picked_columns) > 0, f"No columns found for keypoints {keypoint_names}"
        
        # Verify x, y ordering
        assert (
            picked_columns[::2][0][2] == "x"
            and len(set(map(lambda t: t[2], picked_columns[::2]))) == 1
        ), "X columns not properly ordered"
        assert (
            picked_columns[1::2][0][2] == "y"
            and len(set(map(lambda t: t[2], picked_columns[1::2]))) == 1
        ), "Y columns not properly ordered"
        
        dfs_by_view[view] = df.loc[:, picked_columns]

    # Convert to numpy
    numpy_arrs = {}
    n_keypoints = None
    n_frames = None
    for view in view_names:
        df = dfs_by_view[view]
        n_frames = len(df)
        picked_columns = [c for c in df.columns if c[2] in ("x", "y")]
        n_keypoints = len(picked_columns) // 2  # Number of keypoints (x,y pairs)
        
        nparr = df.to_numpy()  # Shape: (n_frames, n_keypoints * 2)
        # Check for any remaining NaN values
        if np.isnan(nparr).any():
            print(f"  Warning: {view} still has NaN values after filtering!")
            print(f"    NaN count: {np.isnan(nparr).sum()} / {nparr.size}")
        
        # Reshape to (n_frames * n_keypoints, 2) for aniposelib format
        # aniposelib expects (n_cams, n_points, 2) where n_points = n_frames * n_keypoints
        # The reshape organizes as: Frame0_KP0, Frame0_KP1, ..., Frame0_KPN, Frame1_KP0, ...
        nparr = nparr.reshape(n_frames * n_keypoints, 2)
        numpy_arrs[view] = nparr

    # Creates a CxNx2 np array for aniposelib (C=views, N=frames*keypoints, 2)
    p2ds = np.stack([numpy_arrs[v] for v in view_names])
    
    print(f"\nData loaded successfully:")
    print(f"  Views: {p2ds.shape[0]}")
    print(f"  Total points: {p2ds.shape[1]} (frames * keypoints)")
    print(f"  Frames: {n_frames}")
    print(f"  Keypoints: {n_keypoints} ({keypoint_names})")
    
    return p2ds, view_names, keypoint_names


def main():
    parser = argparse.ArgumentParser(
        description='Simple bundle adjustment matching lightning-pose-app implementation'
    )
    parser.add_argument(
        '--csv_files', 
        nargs='+', 
        required=True,
        help='Paths to CSV files (one per view, in order)'
    )
    parser.add_argument(
        '--view_names',
        nargs='+',
        help='View names (optional, will extract from CSV if not provided)'
    )
    parser.add_argument(
        '--calibration_file',
        required=True,
        help='Path to input calibration TOML file'
    )
    parser.add_argument(
        '--output_file',
        required=True,
        help='Path to output calibration TOML file'
    )
    parser.add_argument(
        '--session_filter',
        default=None,
        help='Session name to filter data (e.g., downsampled.2bdf206a)'
    )
    parser.add_argument(
        '--use_iterative',
        action='store_true',
        help='Use bundle_adjust_iter() instead of bundle_adjust() (iterative method)'
    )
    parser.add_argument(
        '--max_iterations',
        type=int,
        default=1000,
        help='Maximum number of iterations for bundle adjustment (default: 1000, matches aniposelib)'
    )
    parser.add_argument(
        '--ftol',
        type=float,
        default=1e-4,
        help='Tolerance for function value convergence (default: 1e-4, matches aniposelib)'
    )
    parser.add_argument(
        '--xtol',
        type=float,
        default=1e-8,
        help='Tolerance for parameter convergence (default: 1e-8)'
    )
    parser.add_argument(
        '--gtol',
        type=float,
        default=1e-8,
        help='Tolerance for gradient convergence (default: 1e-8)'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='linear',
        choices=['linear', 'soft_l1', 'huber', 'cauchy', 'arctan'],
        help='Loss function for bundle adjustment (default: linear). Try "soft_l1" or "huber" for robust optimization if you have outliers.'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=50.0,
        help='Threshold for robust loss functions (default: 50.0 pixels). Only used if loss is not "linear".'
    )
    parser.add_argument(
        '--only_extrinsics',
        action='store_true',
        help='Only optimize camera extrinsics (pose), not intrinsics (focal, distortion, center)'
    )
    parser.add_argument(
        '--keypoints',
        nargs='+',
        default=None,
        help='Keypoint names to use for bundle adjustment (e.g., --keypoints nose paw1). '
             'If not specified, uses all keypoints. Useful when only some keypoints are present '
             'in all frames. The optimized camera parameters will work for all keypoints.'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.csv_files) < 2:
        print("Error: Need at least 2 views for bundle adjustment")
        sys.exit(1)
    
    # Determine view names
    if args.view_names:
        view_names = args.view_names
    else:
        # Extract from CSV files (use filename as view name)
        view_names = [Path(f).stem for f in args.csv_files]
    
    if len(view_names) != len(args.csv_files):
        print(f"Error: Number of view names ({len(view_names)}) doesn't match CSV files ({len(args.csv_files)})")
        sys.exit(1)
    
    print(f"\nView names: {view_names}")
    
    # Load calibration
    print(f"\nLoading calibration from: {args.calibration_file}")
    cg = CameraGroup.load(args.calibration_file)
    calibration_views = list(map(lambda c: c.name, cg.cameras))
    
    # Match CSV files to calibration camera names
    # This handles cases where CSV filenames don't match calibration names
    matched_csv_files = []
    matched_view_names = []
    csv_files_remaining = list(args.csv_files)
    
    def matches_camera(csv_file, camera_name):
        """Check if CSV file name matches camera name (handles left/right variations)"""
        csv_stem = Path(csv_file).stem.lower()
        cam_name_lower = camera_name.lower()
        
        # Exact match
        if cam_name_lower in csv_stem or csv_stem in cam_name_lower:
            return True
        
        # Check for left/right matches
        has_left = 'left' in csv_stem or 'left' in cam_name_lower
        has_right = 'right' in csv_stem or 'right' in cam_name_lower
        
        if has_left and not has_right:
            return 'left' in csv_stem and 'left' in cam_name_lower
        if has_right and not has_left:
            return 'right' in csv_stem and 'right' in cam_name_lower
        
        return False
    
    for calib_view in calibration_views:
        matched = False
        for csv_file in csv_files_remaining:
            if matches_camera(csv_file, calib_view):
                matched_csv_files.append(csv_file)
                matched_view_names.append(calib_view)
                csv_files_remaining.remove(csv_file)
                matched = True
                break
        
        if not matched:
            # If no match found and we have remaining files, use first available
            if csv_files_remaining:
                matched_csv_files.append(csv_files_remaining.pop(0))
                matched_view_names.append(calib_view)
            else:
                raise RuntimeError(
                    f"Could not match calibration camera '{calib_view}' to any CSV file. "
                    f"CSV files: {args.csv_files}, Calibration cameras: {calibration_views}"
                )
    
    if len(matched_csv_files) != len(calibration_views):
        raise RuntimeError(
            f"Mismatch: {len(matched_csv_files)} CSV files matched to {len(calibration_views)} cameras. "
            f"CSV files: {matched_csv_files}, Calibration cameras: {calibration_views}"
        )
    
    print(f"\nMatched CSV files to cameras:")
    for csv_file, view_name in zip(matched_csv_files, matched_view_names):
        print(f"  {view_name}: {csv_file}")
    
    # Load multiview data using matched files and calibration view names
    p2ds, view_names, keypoint_names = load_multiview_data(
        matched_csv_files, 
        matched_view_names,
        session_filter=args.session_filter,
        keypoint_filter=args.keypoints
    )
    
    # Perform bundle adjustment exactly as in lightning-pose-app
    print("\n" + "="*80)
    print("RUNNING BUNDLE ADJUSTMENT")
    print("="*80)
    
    # Triangulate initial 3D points
    print("Step 1: Triangulating initial 3D points...")
    p3ds = cg.triangulate(p2ds)
    
    # Calculate initial reprojection error
    print("Step 2: Calculating initial reprojection error...")
    old_reprojection_error = cg.reprojection_error(p3ds, p2ds)
    old_error_per_cam = np.linalg.norm(old_reprojection_error, axis=2).sum(axis=1)
    
    print(f"  Initial reprojection errors per camera:")
    for i, view in enumerate(view_names):
        print(f"    {view}: {old_error_per_cam[i]:.3f} pixels")
    
    # Perform bundle adjustment
    print("Step 3: Running bundle adjustment...")
    if args.use_iterative:
        print("  Using iterative bundle adjustment (bundle_adjust_iter)")
        print(f"  Parameters: max_nfev={args.max_iterations}, ftol={args.ftol}")
        cg.bundle_adjust_iter(
            p2ds, 
            verbose=True,
            max_nfev=args.max_iterations,
            ftol=args.ftol
        )
    else:
        print("  Using standard bundle adjustment (bundle_adjust)")
        print(f"  Parameters: max_nfev={args.max_iterations}, ftol={args.ftol}, loss={args.loss}")
        if args.loss != 'linear':
            print(f"    threshold={args.threshold} (for robust loss)")
        if args.only_extrinsics:
            print(f"    only_extrinsics=True (only optimizing camera poses, not intrinsics)")
        
        cg.bundle_adjust(
            p2ds, 
            verbose=True,
            max_nfev=args.max_iterations,
            ftol=args.ftol,
            loss=args.loss,
            threshold=args.threshold,
            only_extrinsics=args.only_extrinsics
        )
    
    # Triangulate final 3D points
    print("Step 4: Triangulating final 3D points...")
    p3ds = cg.triangulate(p2ds)
    
    # Calculate final reprojection error
    print("Step 5: Calculating final reprojection error...")
    new_reprojection_error = cg.reprojection_error(p3ds, p2ds)
    new_error_per_cam = np.linalg.norm(new_reprojection_error, axis=2).sum(axis=1)
    
    print(f"  Final reprojection errors per camera:")
    for i, view in enumerate(view_names):
        print(f"    {view}: {new_error_per_cam[i]:.3f} pixels")
    
    # Calculate improvements
    print(f"\nImprovements per camera:")
    for i, view in enumerate(view_names):
        improvement = old_error_per_cam[i] - new_error_per_cam[i]
        improvement_pct = (improvement / old_error_per_cam[i] * 100) if old_error_per_cam[i] > 0 else 0
        print(f"    {view}: {improvement:.3f} pixels ({improvement_pct:.1f}%)")
    
    # Save output
    print(f"\nSaving optimized calibration to: {args.output_file}")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create backup if output file exists
    if output_path.exists():
        backup_path = output_path.parent / f"{output_path.stem}.backup.{int(time.time())}.toml"
        print(f"  Creating backup: {backup_path}")
        output_path.rename(backup_path)
    
    cg.dump(output_path)
    
    print("\n" + "="*80)
    print("BUNDLE ADJUSTMENT COMPLETE!")
    print("="*80)
    print(f"\nOutput file: {args.output_file}")
    print(f"\nResults:")
    print(f"  {'Camera':<15} {'Initial (px)':<15} {'Final (px)':<15} {'Improvement':<15}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    for i, view in enumerate(view_names):
        improvement = old_error_per_cam[i] - new_error_per_cam[i]
        improvement_pct = (improvement / old_error_per_cam[i] * 100) if old_error_per_cam[i] > 0 else 0
        print(f"  {view:<15} {old_error_per_cam[i]:<15.3f} {new_error_per_cam[i]:<15.3f} {improvement:.3f} ({improvement_pct:.1f}%)")


if __name__ == '__main__':
    main()
