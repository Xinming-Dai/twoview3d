"""
Point cloud generation from images and depth maps.
"""

import numpy as np
import cv2
import open3d as o3d
from typing import Literal, Union

InputFormat = Literal["rgb", "bgr", "gray"]

def image_to_depth_estimate(
    image: Union[str, np.ndarray],
    input_format: InputFormat = "bgr",
) -> np.ndarray:
    """
    Convert an image to a depth map estimate.

    The depth map is a 2D array of depth values, where each value is the
    distance from the camera center to the surface. Values are normalized
    to [0, 1] using max scaling.

    Args:
        image: Image as file path (str) or array of shape (H, W) or (H, W, 3).
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


class ImageToPointCloud:
    """
    Build a point cloud from an image file using grayscale as depth estimate.
    """

    def __init__(
        self,
        image_path: str,
        input_format: InputFormat = "bgr",
    ):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        self.input_format = input_format
        self.depth = image_to_depth_estimate(self.image, input_format)
        self.pcd = None

    def to_pcd(self) -> o3d.geometry.PointCloud:
        """
        Convert the depth map to a point cloud.

        Returns:
            pcd: o3d.geometry.PointCloud - the point cloud
        """
        if self.pcd is not None:
            return self.pcd
        points = pseudo_pointcloud_normalized(self.depth)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd = pcd
        return pcd

    def show(self) -> None:
        """
        Display the point cloud with RGB colors from the image.

        Returns:
            None - the point cloud is displayed 
        """
        if self.pcd is None:
            self.to_pcd()
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
        pcd = self.pcd
        pcd.colors = o3d.utility.Vector3dVector(colors)
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
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
        pcd = self.pcd
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])