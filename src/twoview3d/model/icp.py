import copy
from pathlib import Path
from typing import Union

import numpy as np
import open3d as o3d

# Paths to source and target point clouds
SOURCE_PLY = Path(__file__).parent / "frames" / "_iblrig_rightCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0.ply"
TARGET_PLY = Path(__file__).parent / "frames" / "_iblrig_leftCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0.ply"


def draw_registration_result(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transformation: np.ndarray) -> None:
    """Visualize registration result: source (orange) and target (blue) after applying transformation."""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
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
) -> tuple:
    """Load source and target point clouds from PLY files."""
    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(target_path))
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
    if not source.has_normals():
        source.estimate_normals()
    if not target.has_normals():
        target.estimate_normals()

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
