"""
Manually select 5–10 point keypoints between two point clouds for ICP.

Use this as step 1 of the pipeline:
  1. Manually select keypoints (this script)
  2. Estimate initial transform (SVD / Kabsch)
  3. Refine with ICP using point_cloud_registration_with_keypoints.py

Usage:
  python select_keypoints.py [--source SOURCE.ply] [--target TARGET.ply] \\
      [--output keypoints.npz] [--source-color R G B] [--target-color R G B]

  To use colors already stored in the PLY files:
  python select_keypoints.py --source a.ply --target b.ply --use-ply-colors

  To color from images (requires calibration):
  python select_keypoints.py --source a.ply --target b.ply \\
      --source-image left.jpg --target-image right.jpg \\
      --calibration calib.toml --source-cam cam_0 --target-cam cam_1

Source (left) and target (right) are shown side by side in one window.
  - Rotate: left drag
  - Pan: middle drag (or Ctrl + left drag)
  - Zoom: scroll
  - Pick point: Shift + left click
  - Pick 5–10 points on the LEFT cloud first, then the same number on the RIGHT in the same order. Close when done.

Colors: use --source-color/--target-color for a single color, or --source-image/--target-image
  with --calibration and --source-cam/--target-cam to sample colors from images (points in world frame).
"""

import argparse
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import open3d as o3d

try:
    import tomli
except ImportError:
    tomli = None  # type: ignore


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """Load a point cloud from a PLY file."""
    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.has_points():
        raise ValueError(f"No points in {path}")
    return pcd


def _load_calibration(calibration_path: Path) -> dict:
    """Load camera calibration from a TOML file. Returns dict of cam_* -> {matrix, R, translation}."""
    if tomli is None:
        raise ImportError("tomli is required for --calibration. Install with: pip install tomli")
    path = Path(calibration_path)
    with path.open("rb") as f:
        data = tomli.load(f)
    result = {}
    for key, val in data.items():
        if key == "metadata" or not key.startswith("cam_"):
            continue
        matrix = np.array(val["matrix"], dtype=np.float64)
        rvec = np.array(val["rotation"], dtype=np.float64)
        tvec = np.array(val["translation"], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        result[key] = {"matrix": matrix, "R": R, "translation": tvec}
    return result


def colors_from_image(
    points_world: np.ndarray,
    image_path: Path,
    matrix: np.ndarray,
    R: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """
    Project 3D points (world frame) into the image and sample RGB. Returns (N, 3) float in [0, 1].

    Points are transformed to camera frame as p_cam = R.T @ (p_world - translation), then projected.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]

    # World to camera: p_cam = R.T @ (p_world - t)
    p_cam = (R.T @ (points_world - translation).T).T
    x, y, z = p_cam[:, 0], p_cam[:, 1], p_cam[:, 2]

    # Project to pixel coords
    valid = z > 1e-6
    u = np.full_like(z, -1.0)
    v = np.full_like(z, -1.0)
    u[valid] = (fx * x[valid] / z[valid] + cx)
    v[valid] = (fy * y[valid] / z[valid] + cy)

    # Sample image (clamp to bounds)
    u_int = np.clip(np.round(u).astype(int), 0, W - 1)
    v_int = np.clip(np.round(v).astype(int), 0, H - 1)
    # Points behind camera or out of view get a neutral gray
    colors = np.ones((len(points_world), 3), dtype=np.float64) * 0.5
    colors[valid] = img[v_int[valid], u_int[valid]].astype(np.float64) / 255.0
    return colors


def parse_color(s: str) -> list[float]:
    """Parse 'R G B' or 'R,G,B' into three floats in [0,1]. Accepts 0-1 or 0-255."""
    s_strip = s.strip()
    # Detect file paths passed by mistake to --source-color / --target-color
    if "/" in s_strip or "\\" in s_strip or s_strip.endswith((".ply", ".jpg", ".jpeg", ".png", ".exr")):
        raise ValueError(
            f"This looks like a file path, not a color (R G B). "
            "Use --source and --target for point cloud files. "
            "Use --source-color and --target-color only for colors, e.g. --source-color '1 0.6 0'"
        )
    parts = s_strip.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(
            f"Color must be three numbers (R G B), e.g. '1 0.6 0' or '255 153 0'. Got: {s_strip[:80]!r}"
        )
    vals = [float(x) for x in parts]
    if any(v < 0 for v in vals):
        raise ValueError("Color values must be non-negative")
    if all(v <= 1 for v in vals):
        return vals
    if all(v <= 255 for v in vals):
        return [v / 255.0 for v in vals]
    raise ValueError("Use either 0–1 or 0–255 for all components")


def _to_color_array(color: Union[list[float], np.ndarray], n_points: int) -> np.ndarray:
    """Convert uniform color (list of 3) or per-point colors (N,3) to (n_points, 3) float64."""
    if isinstance(color, np.ndarray):
        if color.shape != (n_points, 3):
            raise ValueError(f"Per-point colors must be shape ({n_points}, 3), got {color.shape}")
        return color.astype(np.float64)
    return np.tile(color, (n_points, 1)).astype(np.float64)


def build_side_by_side_cloud(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    source_color: Union[list[float], np.ndarray],
    target_color: Union[list[float], np.ndarray],
    gap_ratio: float = 0.15,
) -> tuple[o3d.geometry.PointCloud, int, float]:
    """
    Build a single point cloud with source on the left and target on the right for side-by-side view.

    source_color / target_color: either [R,G,B] for uniform color or (N,3) array for per-point colors.
    Returns:
        combined_pcd: One point cloud (source points then target points, target translated in x).
        n_source: Number of source points (indices 0 .. n_source-1 are source).
        target_offset_x: Translation applied to target x (so you can recover original target positions).
    """
    src_pts = np.asarray(source_pcd.points)
    tgt_pts = np.asarray(target_pcd.points)
    n_source = len(src_pts)

    src_min_x, src_max_x = src_pts[:, 0].min(), src_pts[:, 0].max()
    tgt_min_x = tgt_pts[:, 0].min()
    extent = max(src_max_x - src_min_x, 1e-6)
    gap = gap_ratio * extent
    target_offset_x = src_max_x - tgt_min_x + gap

    tgt_pts_shifted = tgt_pts + np.array([target_offset_x, 0.0, 0.0])
    combined_pts = np.vstack([src_pts, tgt_pts_shifted])

    source_colors = _to_color_array(source_color, len(src_pts))
    target_colors = _to_color_array(target_color, len(tgt_pts))
    combined_colors = np.vstack([source_colors, target_colors])

    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_pts)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    return combined_pcd, n_source, target_offset_x


def pick_keypoints_side_by_side(
    combined_pcd: o3d.geometry.PointCloud,
    n_source: int,
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    window_title: str,
    min_points: int = 5,
    max_points: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """
    Show combined (side-by-side) point cloud; user picks on left then right. Returns keypoints.

    Picked indices < n_source are source; >= n_source are target (stored in pick order).
    Returns original (un-offset) 3D coordinates for both.
    """
    src_pts = np.asarray(source_pcd.points)
    tgt_pts = np.asarray(target_pcd.points)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_title, width=1400, height=720)
    vis.add_geometry(combined_pcd)
    print(f"\n{window_title}")
    print("  First pick 5–10 points on the LEFT cloud, then the same number on the RIGHT (same order).")
    print("  Shift + left click to pick. Close the window when done.")
    vis.run()
    vis.destroy_window()

    indices = vis.get_picked_points()

    source_pick_indices = [i for i in indices if i < n_source]
    target_pick_indices = [i - n_source for i in indices if i >= n_source]

    n_s, n_t = len(source_pick_indices), len(target_pick_indices)
    if n_s < min_points or n_t < min_points:
        raise ValueError(
            f"Too few points: {n_s} on source (left), {n_t} on target (right). "
            f"Need at least {min_points} on each. Pick on LEFT first, then RIGHT."
        )
    if n_s != n_t:
        raise ValueError(
            f"Different counts: {n_s} on source, {n_t} on target. "
            "Pick the same number on each (left first, then right, in the same order)."
        )
    if n_s > max_points:
        print(f"Warning: {n_s} pairs picked; using first {max_points}.")
        source_pick_indices = source_pick_indices[:max_points]
        target_pick_indices = target_pick_indices[:max_points]

    source_points = src_pts[source_pick_indices]
    target_points = tgt_pts[target_pick_indices]
    return source_points, target_points, source_pick_indices, target_pick_indices


def save_keypoints(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_indices: list[int],
    target_indices: list[int],
    path: Path,
) -> None:
    """Save correspondence arrays to .npz for use in Kabsch and ICP."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        source_points=source_points,
        target_points=target_points,
        source_indices=np.array(source_indices),
        target_indices=np.array(target_indices),
    )
    print(f"Saved {len(source_points)} keypoints to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually select point keypoints between two point clouds for ICP."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to source point cloud (PLY).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Path to target point cloud (PLY).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("keypoints.npz"),
        help="Output path for keypoints (.npz). Default: keypoints.npz",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum number of keypoints (default: 5).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=10,
        help="Maximum number of keypoints (default: 10).",
    )
    parser.add_argument(
        "--source-color",
        type=str,
        default="1 0.6 0",
        help="Source (left) point cloud color as R G B in 0–1 or 0–255 (default: 1 0.6 0 = orange).",
    )
    parser.add_argument(
        "--target-color",
        type=str,
        default="0 0.65 0.93",
        help="Target (right) point cloud color as R G B in 0–1 or 0–255 (default: 0 0.65 0.93 = blue).",
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        default=None,
        help="Image to sample colors from for the source (left) point cloud. Requires --calibration and --source-cam.",
    )
    parser.add_argument(
        "--target-image",
        type=Path,
        default=None,
        help="Image to sample colors from for the target (right) point cloud. Requires --calibration and --target-cam.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Camera calibration TOML (required when using --source-image or --target-image).",
    )
    parser.add_argument(
        "--source-cam",
        type=str,
        default=None,
        help="Camera key in calibration for the source point cloud (e.g. cam_0). Required with --source-image.",
    )
    parser.add_argument(
        "--target-cam",
        type=str,
        default=None,
        help="Camera key in calibration for the target point cloud (e.g. cam_1). Required with --target-image.",
    )
    parser.add_argument(
        "--use-ply-colors",
        action="store_true",
        help="Use vertex colors stored in each PLY file (ignores --source-color/--target-color and image options).",
    )
    args = parser.parse_args()

    # Default paths relative to this script (same layout as point_cloud_registration)
    script_dir = Path(__file__).resolve().parent
    frames_dir = script_dir / "frames"
    default_source = frames_dir / "_iblrig_rightCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0.ply"
    default_target = frames_dir / "_iblrig_leftCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0.ply"

    source_path = args.source or default_source
    target_path = args.target or default_target

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source point cloud not found: {source_path}. Use --source to specify a PLY file."
        )
    if not target_path.exists():
        raise FileNotFoundError(
            f"Target point cloud not found: {target_path}. Use --target to specify a PLY file."
        )

    if args.source_image and not (args.calibration and args.source_cam):
        raise ValueError("--source-image requires --calibration and --source-cam.")
    if args.target_image and not (args.calibration and args.target_cam):
        raise ValueError("--target-image requires --calibration and --target-cam.")

    print("Loading point clouds...")
    source_pcd = load_point_cloud(source_path)
    target_pcd = load_point_cloud(target_path)

    src_pts = np.asarray(source_pcd.points)
    tgt_pts = np.asarray(target_pcd.points)

    calib = _load_calibration(args.calibration) if (args.source_image or args.target_image) else None

    if args.use_ply_colors:
        # Use vertex colors from each PLY; fall back to default uniform if missing
        if source_pcd.has_colors():
            source_color = np.asarray(source_pcd.colors, dtype=np.float64)
            print("Using source point cloud colors from PLY.")
        else:
            source_color = parse_color(args.source_color)
            print("Source PLY has no colors; using uniform color.")
        if target_pcd.has_colors():
            target_color = np.asarray(target_pcd.colors, dtype=np.float64)
            print("Using target point cloud colors from PLY.")
        else:
            target_color = parse_color(args.target_color)
            print("Target PLY has no colors; using uniform color.")
    elif args.source_image:
        cam = calib[args.source_cam]
        source_color = colors_from_image(
            src_pts, args.source_image, cam["matrix"], cam["R"], cam["translation"]
        )
        print(f"Coloring source from image: {args.source_image}")
    else:
        source_color = parse_color(args.source_color)

    if not args.use_ply_colors and args.target_image:
        cam = calib[args.target_cam]
        target_color = colors_from_image(
            tgt_pts, args.target_image, cam["matrix"], cam["R"], cam["translation"]
        )
        print(f"Coloring target from image: {args.target_image}")
    elif not args.use_ply_colors:
        target_color = parse_color(args.target_color)

    combined_pcd, n_source, _ = build_side_by_side_cloud(
        source_pcd, target_pcd, source_color, target_color
    )

    source_points, target_points, source_indices, target_indices = pick_keypoints_side_by_side(
        combined_pcd,
        n_source,
        source_pcd,
        target_pcd,
        "Source (left) & Target (right) — pick keypoints",
        min_points=args.min_points,
        max_points=args.max_points,
    )
    n_corr = len(source_points)
    print(f"Picked {n_corr} keypoints.")

    save_keypoints(
        source_points,
        target_points,
        source_indices,
        target_indices,
        args.output,
    )
    print("Done. Use source_points and target_points from the .npz for Kabsch, then ICP.")


if __name__ == "__main__":
    main()
