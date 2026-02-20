import twoview3d.model.point_cloud as point_cloud
import open3d as o3d
import os

folder_path = "./twoview3d/test/frames/" + "4b00df29-3769-43be-bb40-128b1cba6d35"
calibration_path = "./twoview3d/src/twoview3d/bundle_adjust/_iblrig.downsampled.4b00df29-3769-43be-bb40-128b1cba6d35.toml"

right_vis = os.path.join(folder_path, "_iblrig_rightCamera.downsampled.4b00df29-3769-43be-bb40-128b1cba6d35_vis_t20.jpg")
right_src = os.path.join(folder_path, "_iblrig_rightCamera.downsampled.4b00df29-3769-43be-bb40-128b1cba6d35_src_t20.jpg")
pcd_builder_right = point_cloud.DepthMapImageToPointCloud(
    right_vis,
    calibration_path=calibration_path,
    camera_key="rightCamera",
    depth_scale=5.0,
)

left_vis = os.path.join(
    folder_path,
    "_iblrig_leftCamera.downsampled.4b00df29-3769-43be-bb40-128b1cba6d35_vis_t20__matched_320x256_stretch__ref-_iblrig_rightCamera.downsampled.4b00df29-3769-43be-bb40-128b1cba6d35_src_t20.jpg",
)
left_src = os.path.join(
    folder_path,
    "_iblrig_leftCamera.downsampled.4b00df29-3769-43be-bb40-128b1cba6d35_src_t20__matched_320x256_stretch__ref-_iblrig_rightCamera.downsampled.4b00df29-3769-43be-bb40-128b1cba6d35_src_t20.jpg",
)
pcd_builder_left = point_cloud.DepthMapImageToPointCloud(
    left_vis,
    calibration_path=calibration_path,
    camera_key="leftCamera",
    depth_scale=5.0,
)

pcd_merged = point_cloud.register_two_point_clouds(
    pcd_builder_right,
    pcd_builder_left,
    calibration_path,
    color_path_1=right_src,
    color_path_2=left_src,
)

o3d.visualization.draw_geometries([pcd_merged])