import copy
from pathlib import Path
from typing import Union

import numpy as np
import open3d as o3d

# Paths to source and target point clouds (test/frames/<frame_id>/)
_FRAMES_DIR = Path(__file__).parent.parent.parent.parent / "test" / "frames" / "ecb5520d-1358-434c-95ec-93687ecd1396"
SOURCE_PLY = _FRAMES_DIR / "_iblrig_rightCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0_removed_background.ply"
TARGET_PLY = _FRAMES_DIR / "_iblrig_leftCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0_removed_background.ply"


def draw_registration_result(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    use_original_colors: bool = True,
) -> None:
    """Visualize registration result after applying transformation.

    Parameters
    ----------
    use_original_colors : bool
        If True, preserve original point cloud colors. If False, paint source
        orange and target blue for clearer distinction.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if not use_original_colors:
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_plotly([source_temp, target_temp])


def evaluate_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    threshold: float = 0.02,
) -> o3d.pipelines.registration.RegistrationResult:
    """Evaluate registration quality (fitness and inlier_rmse)."""
    return o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, transformation
    )


def run_point_to_point_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    trans_init: np.ndarray,
    threshold: float = 0.02,
) -> o3d.pipelines.registration.RegistrationResult:
    """Run point-to-point ICP registration."""
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )


def run_point_to_plane_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    trans_init: np.ndarray,
    threshold: float = 0.02,
) -> o3d.pipelines.registration.RegistrationResult:
    """Run point-to-plane ICP registration (requires normals)."""
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )


def load_point_clouds(
    source_path: Union[Path, str],
    target_path: Union[Path, str],
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """Load source and target point clouds from PLY files."""
    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(target_path))
    if len(np.asarray(source.points)) == 0:
        raise FileNotFoundError(
            f"Failed to load source point cloud from {source_path}. "
            "Check that the file exists and is a valid PLY."
        )
    if len(np.asarray(target.points)) == 0:
        raise FileNotFoundError(
            f"Failed to load target point cloud from {target_path}. "
            "Check that the file exists and is a valid PLY."
        )
    return source, target


def main() -> None:
    source_path = SOURCE_PLY
    target_path = TARGET_PLY

    print(f"Loading source: {source_path}")
    print(f"Loading target: {target_path}")
    source, target = load_point_clouds(source_path, target_path)

    # Initial transformation (identity for unknown prior alignment)
    trans_init = np.eye(4)

    threshold = 0.02

    # Initial alignment evaluation
    print("\n--- Initial alignment ---")
    evaluation = evaluate_registration(source, target, trans_init, threshold)
    print(evaluation)

    # Point-to-point ICP
    print("\n--- Point-to-point ICP ---")
    reg_p2p = run_point_to_point_icp(source, target, trans_init, threshold)
    print(reg_p2p)
    print("Transformation:")
    print(reg_p2p.transformation)
    evaluation_p2p = evaluate_registration(source, target, reg_p2p.transformation, threshold)
    print("Evaluation after P2P:", evaluation_p2p)

    # Point-to-plane ICP (requires normals - estimate if missing)
    # Depth-derived point clouds typically have no normals; estimate from neighbors
    if not source.has_normals():
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    if not target.has_normals():
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

    print("\n--- Point-to-plane ICP ---")
    reg_p2plane = run_point_to_plane_icp(source, target, reg_p2p.transformation, threshold)
    print(reg_p2plane)
    print("Transformation:")
    print(reg_p2plane.transformation)
    evaluation_p2plane = evaluate_registration(source, target, reg_p2plane.transformation, threshold)
    print("Evaluation after P2Plane:", evaluation_p2plane)

    # Visualize final result
    print("\nVisualizing registration result...")
    draw_registration_result(source, target, reg_p2plane.transformation)


if __name__ == "__main__":
    main()
