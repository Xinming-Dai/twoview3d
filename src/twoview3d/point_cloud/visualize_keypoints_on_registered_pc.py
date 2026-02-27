"""
Visualize keypoints on the registered point cloud.

The registered point cloud is built by icp_with_keypoints.py:
  1. Transforming the SOURCE point cloud (--source) into the TARGET frame (--target)
  2. Merging: [transformed_source_points, target_points]
  All points in the registered PLY are in the TARGET frame.

Keypoints in the .npz file follow the same convention as your icp_with_keypoints.py call:
  - source_points: 3D points from the SOURCE point cloud (--source), need transform to target frame
  - target_points: 3D points from the TARGET point cloud (--target), already in target/registered frame

Example: if you used --source left.ply --target right.ply, then
  source_points = left camera, target_points = right camera.

Usage:
  python visualize_keypoints_on_registered_pc.py

  Or with custom paths:
  python visualize_keypoints_on_registered_pc.py \
      --registered-ply path/to/registered.ply \
      --keypoints path/to/keypoints.npz \
      --transform path/to/transform.npy
"""

from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


def load_keypoints(
    keypoint_path: Path,
    source_ply: Optional[Path] = None,
    target_ply: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load keypoints from .npz. Returns (source_points, target_points) as (N,3) arrays.

    If source_points/target_points exist, use them directly.
    If only source_indices/target_indices exist, extract 3D points from the given PLY files.

    Arguments:
        path: Path to the .npz file containing keypoints.
        source_ply: Path to the source PLY file.
        target_ply: Path to the target PLY file.

    Returns:
        source_points: 3D points from the source point cloud.
        target_points: 3D points from the target point cloud.
    """
    keypoints = np.load(keypoint_path)
    if "source_points" in keypoints and "target_points" in keypoints:
        return keypoints["source_points"], keypoints["target_points"]
    if "source_indices" in keypoints and "target_indices" in keypoints:
        if source_ply is None or target_ply is None:
            raise ValueError(
                "keypoints npz has indices but not 3D coordinates. "
                "Pass --source-ply and --target-ply (original point clouds) to extract points by index."
            )
        src_pcd = o3d.io.read_point_cloud(str(source_ply))
        tgt_pcd = o3d.io.read_point_cloud(str(target_ply))
        src_pts = np.asarray(src_pcd.points)[keypoints["source_indices"]]
        tgt_pts = np.asarray(tgt_pcd.points)[keypoints["target_indices"]]
        return src_pts, tgt_pts
    raise KeyError(f"Expected source_points/target_points or source_indices/target_indices in {keypoint_path}. Keys: {list(keypoints.keys())}")


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 SE(3) transform to (N,3) points. Returns (N,3).
    
    Arguments:
        points: (N,3) array of points.
        T: 4x4 SE(3) transform matrix.

    Returns:
        transformed: (N,3) array of transformed points.
    """
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    pts_h = np.hstack([pts, ones])  # (N, 4)
    transformed = (T @ pts_h.T).T  # (N, 4)
    return transformed[:, :3]


def create_sphere(center: np.ndarray, radius: float = 0.005, color: tuple = (1, 0, 0)) -> o3d.geometry.TriangleMesh:
    """Create a small sphere mesh for visualizing a keypoint.
    
    Arguments:
        center: (3,) array of center coordinates.
        radius: Radius of the sphere in metres.
        color: Color of the sphere as R G B 0-1.

    Returns:
        sphere: o3d.geometry.TriangleMesh of the sphere.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Visualize keypoints on the registered point cloud.")
    frame_dir = Path(__file__).parent / "frames" / "ecb5520d-1358-434c-95ec-93687ecd1396"
    parser.add_argument(
        "--registered-ply",
        type=Path,
        default=frame_dir / "_iblrig_downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0_registered_pc_removed_background_original_color.ply",
        help="Path to the registered (merged) point cloud PLY.",
    )
    parser.add_argument(
        "--keypoints",
        type=Path,
        default=frame_dir / "keypoints_removed_bg.npz",
        help="Path to keypoints .npz (source_points, target_points or source_indices, target_indices).",
    )
    parser.add_argument(
        "--source-ply",
        type=Path,
        default=frame_dir / "_iblrig_leftCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0_removed_background_original_color.ply",
        help="Original source point cloud (same as --source in icp_with_keypoints). Required if keypoints only have indices.",
    )
    parser.add_argument(
        "--target-ply",
        type=Path,
        default=frame_dir / "_iblrig_rightCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0_removed_background_original_color.ply",
        help="Original target point cloud (same as --target in icp_with_keypoints). Required if keypoints only have indices.",
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=frame_dir / "transform_removed_background_original_color.npy",
        help="Path to the 4x4 transform (source→target frame) .npy.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.02,
        help="Radius of keypoint spheres in metres (default: 0.02).",
    )
    parser.add_argument(
        "--source-color",
        type=str,
        default="1 0.6 0",
        help="Color for source (right) keypoints as R G B 0-1 (default: orange).",
    )
    parser.add_argument(
        "--target-color",
        type=str,
        default="0 0.65 0.93",
        help="Color for target (left) keypoints as R G B 0-1 (default: blue).",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading registered point cloud: {args.registered_ply}")
    pcd = o3d.io.read_point_cloud(str(args.registered_ply))
    if len(pcd.points) == 0:
        raise FileNotFoundError(f"Empty or invalid point cloud: {args.registered_ply}")

    print(f"Loading keypoints: {args.keypoints}")
    source_pts, target_pts = load_keypoints(
        args.keypoints,
        source_ply=args.source_ply,
        target_ply=args.target_ply,
    )
    n_kp = len(source_pts)
    print(f"  Found {n_kp} keypoint pairs")

    print(f"Loading transform: {args.transform}")
    T = np.load(args.transform)
    if T.shape != (4, 4):
        raise ValueError(f"Transform must be 4x4, got shape {T.shape}")

    # Transform source keypoints to target (registered) frame
    source_pts_reg = transform_points(source_pts, T)
    # target_pts are already in target frame

    # Parse colors
    def parse_color(s: str) -> tuple:
        parts = [float(x) for x in s.split()]
        if len(parts) == 3:
            if all(0 <= x <= 1 for x in parts):
                return tuple(parts)
            return tuple(x / 255 for x in parts)
        raise ValueError(f"Color must be 'R G B', got: {s}")

    src_color = parse_color(args.source_color)
    tgt_color = parse_color(args.target_color)

    # Build geometry list: point cloud + spheres for each keypoint
    geometries = [pcd]

    for i in range(n_kp):
        # Source keypoint (transformed to registered frame)
        s = create_sphere(source_pts_reg[i], radius=args.radius, color=src_color)
        geometries.append(s)
        # Target keypoint (already in registered frame)
        t = create_sphere(target_pts[i], radius=args.radius, color=tgt_color)
        geometries.append(t)

    print(f"\nVisualizing: registered point cloud + {n_kp} source (orange) + {n_kp} target (blue) keypoints")
    print("  Source keypoints: transformed from right camera to left (registered) frame")
    print("  Target keypoints: from left camera, already in registered frame")
    print("  Close the window when done.")
    # Use draw_geometries instead of draw_plotly - Plotly often renders mesh colors as grey
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    main()
