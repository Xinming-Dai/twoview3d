"""
Kabsch + ICP pipeline for point cloud registration.

run select_correspondences.py first to interactively pick and save landmark correspondences, then run this script to compute the transform.

Usage:  
    python icp_with_keypoints.py --source path/to/source.ply --target path/to/target.ply --correspondences path/to/correspondences.npz --output path/to/transform.npy
    python icp_with_keypoints.py ... --save-ply path/to/registered.ply   # save merged point cloud with original colors

""" 

from pathlib import Path
import copy
import numpy as np
import open3d as o3d

from twoview3d.model.icp import (
    draw_registration_result,
    evaluate_registration,
    load_point_clouds,
    run_point_to_plane_icp,
    run_point_to_point_icp,
)

def pick_points(pcd: o3d.geometry.PointCloud, window_name: str = "Pick Points") -> list[int]:
    """
    Open an interactive viewer and let the user select points.

    Controls:
        Shift + left-click   : select a point
        Shift + right-click  : deselect the last point
        Q / Escape           : confirm selection and close window

    Returns
    -------
    list[int]
        Indices of the selected points in *pcd*.
    """
    print(f"\n[{window_name}]")
    print("  Shift + left-click  : select a point")
    print("  Shift + right-click : deselect last")
    print("  Q / Escape          : confirm and close\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    indices = vis.get_picked_points()
    print(f"  → {len(indices)} point(s) selected: {indices}")
    return indices


def pick_correspondences(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
) -> tuple[list[int], list[int]]:
    """
    Interactively pick matching landmark pairs from two point clouds.

    Pick landmarks in the **source** cloud first, then pick the same
    landmarks (in the same order) in the **target** cloud.

    Returns
    -------
    source_indices, target_indices : list[int], list[int]
        Parallel lists of point indices — ``source_indices[i]`` corresponds
        to ``target_indices[i]``.

    Raises
    ------
    ValueError
        If the two selections have different lengths, or fewer than 3 pairs.
    """
    print("\n=== Correspondence Selection ===")
    print("Pick the same anatomical landmarks in BOTH clouds, in the same order.\n")

    source_indices = pick_points(source_pcd, window_name="SOURCE — pick landmarks (Shift+click)")
    target_indices = pick_points(target_pcd, window_name="TARGET — pick matching landmarks (Shift+click)")

    n_src, n_tgt = len(source_indices), len(target_indices)
    if n_src != n_tgt:
        raise ValueError(
            f"Selection count mismatch: source has {n_src} point(s), "
            f"target has {n_tgt}. Pick the same number in each window."
        )
    if n_src < 3:
        raise ValueError(
            f"At least 3 correspondences are required for a unique rigid transform; got {n_src}."
        )

    print(f"\n{n_src} correspondence pair(s) confirmed.")
    return source_indices, target_indices

def kabsch_transform(src_pts: np.ndarray, tgt_pts: np.ndarray) -> np.ndarray:
    """
    Estimate a rigid-body transform (R, t) from N≥3 point correspondences
    using the Kabsch algorithm (SVD-based least-squares).

    Parameters
    ----------
    src_pts : (N, 3) array
        Points in the source frame.
    tgt_pts : (N, 3) array
        Corresponding points in the target frame.

    Returns
    -------
    T : (4, 4) ndarray
        SE(3) transformation matrix such that ``tgt ≈ T @ [src | 1]ᵀ``.
    """
    src = np.asarray(src_pts, dtype=float)
    tgt = np.asarray(tgt_pts, dtype=float)

    # Centre both point sets
    src_c = src.mean(axis=0)
    tgt_c = tgt.mean(axis=0)
    src_demean = src - src_c
    tgt_demean = tgt - tgt_c

    # Cross-covariance matrix
    H = src_demean.T @ tgt_demean  # (3, 3)

    U, _, Vt = np.linalg.svd(H)

    # Correct for reflection (ensure det(R) = +1)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(Vt.T @ U.T)

    R = Vt.T @ D @ U.T
    t = tgt_c - R @ src_c

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def estimate_initial_transform(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    source_indices: list[int],
    target_indices: list[int],
) -> np.ndarray:
    """
    Compute an initial SE(3) transform from manually selected correspondences
    using the Kabsch algorithm.

    Returns
    -------
    T_init : (4, 4) ndarray
    """
    src_pts = np.asarray(source_pcd.points)[source_indices]
    tgt_pts = np.asarray(target_pcd.points)[target_indices]

    T_init = kabsch_transform(src_pts, tgt_pts)

    # Per-point residuals after applying the estimated transform
    src_transformed = (T_init[:3, :3] @ src_pts.T).T + T_init[:3, 3]
    residuals = np.linalg.norm(src_transformed - tgt_pts, axis=1)
    print(
        f"\nKabsch residuals (metres):  "
        f"mean={residuals.mean():.4f}  "
        f"max={residuals.max():.4f}"
    )
    return T_init

def refine_with_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    max_correspondence_distance: float = 0.05,
) -> tuple[o3d.pipelines.registration.RegistrationResult, np.ndarray]:
    """
    Refine the Kabsch initial transform with ICP.

    Uses point-to-plane ICP when the target already has normals, otherwise
    falls back to point-to-point ICP.

    Parameters
    ----------
    max_correspondence_distance : float
        Maximum distance (in the same units as the point cloud) to consider
        two points as correspondences. Default 0.05 m.

    Returns
    -------
    result : o3d.pipelines.registration.RegistrationResult
    T_icp  : (4, 4) ndarray — refined transformation matrix
    """
    use_plane = target_pcd.has_normals() and len(target_pcd.normals) == len(target_pcd.points)

    if use_plane:
        print("\nUsing point-to-plane ICP (normals found on target).")
        result = run_point_to_plane_icp(source_pcd, target_pcd, T_init, max_correspondence_distance)
    else:
        print("\nUsing point-to-point ICP (no normals on target).")
        result = run_point_to_point_icp(source_pcd, target_pcd, T_init, max_correspondence_distance)

    T_icp = np.asarray(result.transformation)
    print(f"ICP result:  fitness={result.fitness:.4f}  inlier_rmse={result.inlier_rmse:.6f}")
    return result, T_icp


def save_registered_point_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save the registered (aligned) source and target point clouds merged into one PLY,
    preserving the original colors from each point cloud.
    """
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(transformation)

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(source_copy.points), np.asarray(target_copy.points)])
    )

    # Preserve original colors; use gray if a cloud has no colors
    src_colors = np.asarray(source_copy.colors) if source_copy.has_colors() else np.ones((len(source_copy.points), 3)) * 0.5
    tgt_colors = np.asarray(target_copy.colors) if target_copy.has_colors() else np.ones((len(target_copy.points), 3)) * 0.5
    merged.colors = o3d.utility.Vector3dVector(np.vstack([src_colors, tgt_colors]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), merged)
    print(f"Registered point cloud (original colors) saved to {output_path}")


def run_manual_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    max_correspondence_distance: float = 0.05,
    show_result: bool = True,
    use_original_colors: bool = True,
) -> tuple[np.ndarray, o3d.pipelines.registration.RegistrationResult]:
    """
    Full manual-ICP pipeline:

    1. Interactively pick N≥3 corresponding landmark pairs.
    2. Estimate an initial SE(3) transform via the Kabsch algorithm.
    3. Refine the transform with ICP.

    Parameters
    ----------
    source_pcd                 : Open3D PointCloud (to be aligned)
    target_pcd                 : Open3D PointCloud (reference)
    max_correspondence_distance: ICP correspondence distance threshold
    show_result                : Display the aligned clouds when done

    Returns
    -------
    T_icp  : (4, 4) ndarray — final source→target transformation
    result : o3d.pipelines.registration.RegistrationResult
    """
    # Step 1
    src_idx, tgt_idx = pick_correspondences(source_pcd, target_pcd)

    # Step 2
    print("\n--- Kabsch initial transform ---")
    T_init = estimate_initial_transform(source_pcd, target_pcd, src_idx, tgt_idx)

    # Step 3
    print("\n--- ICP refinement ---")
    result, T_icp = refine_with_icp(source_pcd, target_pcd, T_init, max_correspondence_distance)

    if show_result:
        draw_registration_result(source_pcd, target_pcd, T_icp, use_original_colors=use_original_colors)

    return T_icp, result

SOURCE_PLY = (
    Path(__file__).parent
    / "frames"
    / "_iblrig_rightCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0.ply"
)
TARGET_PLY = (
    Path(__file__).parent
    / "frames"
    / "_iblrig_leftCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0.ply"
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Kabsch + ICP from saved or interactively-picked correspondences."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_PLY, help="Source PLY file.")
    parser.add_argument("--target", type=Path, default=TARGET_PLY, help="Target PLY file.")
    parser.add_argument(
        "--correspondences",
        type=Path,
        default=None,
        help="Path to a .npz saved by select_correspondences.py. Skips interactive picking.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.05,
        help="ICP max correspondence distance (default: 0.05).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save the final 4×4 transform to a .npy file.",
    )
    parser.add_argument(
        "--save-ply",
        type=Path,
        default=None,
        help="Save the registered (merged) point clouds to a PLY file with original colors.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the registration result window.",
    )
    parser.add_argument(
        "--no-original-colors",
        action="store_true",
        help="Use blue (target) and orange (source) instead of original point cloud colors when displaying.",
    )
    args = parser.parse_args()

    print(f"Loading source: {args.source}")
    print(f"Loading target: {args.target}")
    source, target = load_point_clouds(args.source, args.target)

    if args.correspondences is not None:
        data = np.load(args.correspondences)
        src_idx = data["source_indices"].tolist()
        tgt_idx = data["target_indices"].tolist()
        print(f"Loaded {len(src_idx)} correspondences from {args.correspondences}")

        print("\n--- Kabsch initial transform ---")
        T_init = estimate_initial_transform(source, target, src_idx, tgt_idx)

        print("\n--- ICP refinement ---")
        _, T_icp = refine_with_icp(source, target, T_init, args.max_distance)

        if not args.no_show:
            draw_registration_result(source, target, T_icp, use_original_colors=not args.no_original_colors)
    else:
        T_icp, _ = run_manual_icp(
            source, target,
            max_correspondence_distance=args.max_distance,
            show_result=not args.no_show,
            use_original_colors=not args.no_original_colors,
        )

    print("\n--- Final transform ---")
    print(T_icp)

    eval_final = evaluate_registration(source, target, T_icp, threshold=0.02)
    print("\n--- Final evaluation ---")
    print(eval_final)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, T_icp)
        print(f"Transform saved to {args.output}")

    if args.save_ply is not None:
        save_registered_point_clouds(source, target, T_icp, args.save_ply)


if __name__ == "__main__":
    main()
