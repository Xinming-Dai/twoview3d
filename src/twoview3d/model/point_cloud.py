"""
Point cloud generation from images and depth maps.
"""

from pathlib import Path
import tomli
import numpy as np
import cv2
import open3d as o3d
from typing import Literal, Union


InputFormat = Literal["rgb", "bgr", "gray"]


def load_calibration(calibration_path: Union[str, Path]) -> dict:
    """
    Load camera calibration from a TOML file.

    Args:
        calibration_path: Path to calibration.toml.

    Returns:
        Dict mapping camera keys (e.g. "cam_0", "cam_1") to camera params:
        - name: str
        - size: (width, height)
        - matrix: 3x3 numpy array (intrinsics K)
        - distortions: 5-element array
        - rotation: 3-element Rodrigues vector
        - translation: 3-element vector
        - R: 3x3 rotation matrix (from Rodrigues)
    """
    path = Path(calibration_path)
    with path.open("rb") as f:
        data = tomli.load(f)

    result = {}
    for key, val in data.items():
        if key == "metadata" or not key.startswith("cam_"):
            continue
        matrix = np.array(val["matrix"], dtype=np.float64)
        distortions = np.array(val["distortions"], dtype=np.float64)
        rvec = np.array(val["rotation"], dtype=np.float64)
        tvec = np.array(val["translation"], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        result[key] = {
            "name": val["name"],
            "size": tuple(val["size"]),
            "matrix": matrix,
            "distortions": distortions,
            "rotation": rvec,
            "translation": tvec,
            "R": R,
        }
    return result


def depth_to_pointcloud_calibrated(
    depth: np.ndarray,
    matrix: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    distortions: np.ndarray | None = None,
    depth_scale: float = 1.0,
    depth_max: float | None = None,
    to_world_frame: bool = True,
) -> np.ndarray:
    """
    Back-project a depth map to 3D points using camera intrinsics and extrinsics.

    For pixel (u, v) with depth d:
        x_cam = (u - cx) * d / fx
        y_cam = (v - cy) * d / fy
        z_cam = d
    Then optionally: world_point = R @ camera_point + t

    Args:
        depth: Shape (H, W), depth per pixel (any units; scaled by depth_scale).
        matrix: 3x3 camera intrinsic matrix K.
        rotation: 3x3 rotation matrix (camera to world).
        translation: 3-element translation vector (camera position in world).
        distortions: Optional 5-element distortion coeffs; if provided, pixels
            are undistorted before back-projection.
        depth_scale: Scale factor for depth (e.g. to convert [0,1] to meters).
        depth_max: If set, points with depth > depth_max are dropped.
        to_world_frame: If True, transform points to world frame; else keep in
            camera frame.

    Returns:
        points: Shape (N, 3), each row is (x, y, z).
    """
    H, W = depth.shape
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]

    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    if distortions is not None and np.any(distortions != 0):
        pts = np.stack([u.ravel(), v.ravel()], axis=1)
        pts_undist = cv2.undistortPoints(
            pts.reshape(1, -1, 2), matrix, distortions, P=matrix
        ).reshape(-1, 2)
        u = pts_undist[:, 0].reshape(H, W)
        v = pts_undist[:, 1].reshape(H, W)

    d = depth.astype(np.float64).ravel() * depth_scale
    valid = d > 0
    if depth_max is not None:
        valid &= d <= depth_max

    x_cam = (u.ravel() - cx) * d / fx
    y_cam = (v.ravel() - cy) * d / fy
    z_cam = d

    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    points_cam = points_cam[valid]

    if to_world_frame:
        points = (rotation @ points_cam.T).T + translation
    else:
        points = points_cam

    return points.astype(np.float32)

def depth_map_to_depth_estimate(
    image: Union[str, np.ndarray],
    input_format: InputFormat = "bgr",
) -> np.ndarray:
    """
    Convert a depth map image to a depth map estimate. For example, the depth map image can be a video-depth-anything frame.

    The depth map is a 2D array of depth values, where each value is the
    distance from the camera center to the surface. Values are normalized
    to [0, 1] using max scaling.

    Args:
        image: Depth map image as file path (str) or array of shape (H, W).
        input_format: "rgb", "bgr", or "gray". For rgb/bgr, grayscale is
            computed via Gray = 0.299*R + 0.587*G + 0.114*B.

    Returns:
        depth: Shape (H, W), dtype float32, values in [0, 1].
    """
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image}")

    if input_format == "bgr":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif input_format == "rgb":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif input_format == "gray":
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Invalid input format: {input_format}")

    depth = gray.astype(np.float32)
    depth = depth / depth.max()
    return depth


def pseudo_pointcloud_normalized(depth: np.ndarray) -> np.ndarray:
    """
    Convert a depth map to a pseudo point cloud in normalized coordinates.

    The origin is the center of the image; X and Y range is [-0.5, 0.5].
    Z is the depth value. Y is flipped to align with 3D convention.

    Args:
        depth: Shape (H, W), depth values per pixel.

    Returns:
        points: Shape (H*W, 3), each row is (x, y, z).
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - W / 2) / W
    Y = (v - H / 2) / H
    Z = depth
    points = np.stack([X, -Y, Z], axis=-1)
    return points.reshape(-1, 3)


class DepthMapImageToPointCloud:
    """
    Build a point cloud from a depth map image file using grayscale as depth estimate.

    Args:
        depth_map_image_path: Path to the depth map image.
        input_format: "rgb", "bgr", or "gray". For rgb/bgr, grayscale is
            computed via Gray = 0.299*R + 0.587*G + 0.114*B.
        calibration_path: Optional path to calibration.toml. If provided,
            uses camera intrinsics and extrinsics for metric back-projection.
        camera_key: Camera key in calibration (e.g. "cam_0", "cam_1") or
            camera name (e.g. "rightCamera"). Required if calibration_path is set.
        depth_scale: Scale factor for depth. When depth is in [0, 1], set to
            max depth in meters (e.g. 5.0) for metric output. Ignored if no
            calibration.

    Returns:
        DepthMapImageToPointCloud object
    """

    def __init__(
        self,
        depth_map_image_path: str,
        input_format: InputFormat = "bgr",
        calibration_path: Union[str, Path, None] = None,
        camera_key: str | None = None,
        depth_scale: float = 1.0,
    ):
        self.depth_map_image_path = depth_map_image_path
        self.depth_map_image = cv2.imread(depth_map_image_path)
        if self.depth_map_image is None:
            raise FileNotFoundError(f"Cannot load image: {depth_map_image_path}")
        self.input_format = input_format
        self.depth = depth_map_to_depth_estimate(self.depth_map_image, input_format)
        self.pcd = None

        self._calib = None
        self._cam_params = None
        if calibration_path is not None:
            if camera_key is None:
                raise ValueError("camera_key is required when calibration_path is set")
            self._calib = load_calibration(calibration_path)
            # Resolve by key (cam_0) or by name (rightCamera)
            if camera_key in self._calib:
                self._cam_params = self._calib[camera_key]
            else:
                for c in self._calib.values():
                    if c["name"] == camera_key:
                        self._cam_params = c
                        break
                if self._cam_params is None:
                    raise ValueError(
                        f"Camera '{camera_key}' not found in calibration. "
                        f"Available: {list(self._calib.keys())} "
                        f"(names: {[c['name'] for c in self._calib.values()]})"
                    )
            self._depth_scale = depth_scale
        else:
            self._depth_scale = 1.0

    def to_pcd(self) -> o3d.geometry.PointCloud:
        """
        Convert the depth map to a point cloud.

        Uses calibration intrinsics/extrinsics if provided; otherwise uses
        normalized pseudo coordinates.

        Returns:
            pcd: o3d.geometry.PointCloud - the point cloud
        """
        if self.pcd is not None:
            return self.pcd
        if self._cam_params is not None:
            points = depth_to_pointcloud_calibrated(
                self.depth,
                matrix=self._cam_params["matrix"],
                rotation=self._cam_params["R"],
                translation=self._cam_params["translation"],
                distortions=self._cam_params["distortions"],
                depth_scale=self._depth_scale,
                to_world_frame=True,
            )
        else:
            points = pseudo_pointcloud_normalized(self.depth)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd = pcd
        return pcd

    def _image_to_point_colors(self, image: np.ndarray) -> np.ndarray:
        """
        Convert a BGR image to point colors for the point cloud.

        Args:
            image: BGR image of shape (H, W, 3), same size as depth map.

        Returns:
            colors: Shape (N, 3), float32 in [0, 1], filtered by valid depth
                if calibration is used.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
        if self._cam_params is not None:
            valid = self.depth.ravel() > 0
            colors = colors[valid]
        return colors

    def show(self) -> None:
        """
        Display the point cloud with RGB colors from the image.

        Returns:
            None - the point cloud is displayed 
        """
        if self.pcd is None:
            self.to_pcd()
        pcd = self.pcd
        pcd.colors = o3d.utility.Vector3dVector(
            self._image_to_point_colors(self.depth_map_image)
        )
        o3d.visualization.draw_geometries([pcd])
    
    def show_in_original_color(self, image_path: str) -> None:
        """
        Display the point cloud with RGB colors from the original image.

        Args:
            image_path: Path to the color image. Must have the same dimensions
                (H, W) as the image used to build this point cloud.

        Returns:
            None - the point cloud is displayed
        """
        if self.pcd is None:
            self.to_pcd()
        color_image = cv2.imread(image_path)
        if color_image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        if color_image.shape[:2] != self.depth.shape:
            raise ValueError(
                f"Color image shape {color_image.shape[:2]} does not match "
                f"point cloud depth shape {self.depth.shape}"
            )
        pcd = self.pcd
        pcd.colors = o3d.utility.Vector3dVector(
            self._image_to_point_colors(color_image)
        )
        o3d.visualization.draw_geometries([pcd])

    def save_ply(self, color_image_path: str | None = None) -> str:
        """
        Save the point cloud to a PLY file.

        The output path is the same as the depth map image path, with the
        file extension changed to .ply.

        Args:
            color_image_path: Optional path to a color image for point colors.
                If provided, must have the same dimensions as the depth map.
                If None, uses colors from the depth map image.

        Returns:
            The path where the PLY file was saved.
        """
        if self.pcd is None:
            self.to_pcd()
        save_path = Path(self.depth_map_image_path).with_suffix(".ply")

        pcd = self.pcd
        if color_image_path is not None:
            color_image = cv2.imread(color_image_path)
            if color_image is None:
                raise FileNotFoundError(f"Cannot load image: {color_image_path}")
            if color_image.shape[:2] != self.depth.shape:
                raise ValueError(
                    f"Color image shape {color_image.shape[:2]} does not match "
                    f"point cloud depth shape {self.depth.shape}"
                )
            image = color_image
        else:
            image = self.depth_map_image
        pcd.colors = o3d.utility.Vector3dVector(
            self._image_to_point_colors(image)
        )

        o3d.io.write_point_cloud(str(save_path), pcd)
        return str(save_path)
