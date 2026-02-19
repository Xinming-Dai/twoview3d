# This script is for fusing two point clouds using camera parameters using general camera parameters
# /Users/dai/Documents/Phd/Research/twoview3d/code/twoview3d/src/twoview3d/data/calibration.toml
import twoview3d.model.point_cloud as point_cloud
import os


folder_path = "./twoview3d/test/frames/" + "ecb5520d-1358-434c-95ec-93687ecd1396"
image_path = os.path.join(folder_path, "_iblrig_rightCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0.jpg")
original_image_path = os.path.join(folder_path, "_iblrig_rightCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_src_t17p0.jpg")

pc = point_cloud.DepthMapImageToPointCloud(image_path, input_format="bgr")
pc.show_in_original_color(original_image_path)
# ply_path = pc.save_ply()
# print(f"Point cloud saved to {ply_path}")


calibration_path = "./twoview3d/src/twoview3d/bundle_adjust/_iblrig.downsampled.4b00df29-3769-43be-bb40-128b1cba6d35.toml"
pcd_builder_right = point_cloud.DepthMapImageToPointCloud(
    image_path,
    calibration_path=calibration_path,
    camera_key="rightCamera",  # or "rightCamera"
    depth_scale=5.0,     # if depth is [0,1], scale to meters
)
pcd = pcd_builder_right.to_pcd()
pcd_builder_right.show_in_original_color(original_image_path)


folder_path = "./twoview3d/test/frames/" + "ecb5520d-1358-434c-95ec-93687ecd1396"
image_path = os.path.join(folder_path, "_iblrig_leftCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0.jpg")
original_image_path = os.path.join(folder_path, "_iblrig_leftCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_src_t17p0.jpg")
pcd_builder_left = point_cloud.DepthMapImageToPointCloud(
    image_path,
    calibration_path="./twoview3d/src/twoview3d/bundle_adjust/_iblrig.downsampled.4b00df29-3769-43be-bb40-128b1cba6d35.toml",
    camera_key="leftCamera",  # or "rightCamera"
    depth_scale=5.0,     # if depth is [0,1], scale to meters
)
pcd = pcd_builder_left.to_pcd()
pcd_builder_left.show_in_original_color(original_image_path)