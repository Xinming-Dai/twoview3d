from twoview3d.model.point_cloud import (
    interactive_depth_picker,
    remove_background_by_depth,
)
import os
from twoview3d.model import point_cloud

folder_path = "./twoview3d/test/frames/" + "ecb5520d-1358-434c-95ec-93687ecd1396"
right_vis = os.path.join(folder_path, "_iblrig_rightCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0.jpg")
original_right_vis = os.path.join(folder_path, "_iblrig_rightCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_src_t17p0.jpg")

# Build point cloud with background removed (builder-based workflow)
pcd_builder_right = point_cloud.DepthMapImageToPointCloud(right_vis, input_format="bgr")
# threshold = pcd_builder_right.interactive_pick_threshold()  # or interactive_depth_picker(right_vis)
threshold = 0.15
if threshold is not None:
    pcd_builder_right.remove_background(threshold)
pcd_builder_right.to_pcd()
pcd_builder_right.show()
# pcd_builder_right.save_ply(color_image_path=original_right_vis)



# folder_path = "./twoview3d/test/frames/" + "ecb5520d-1358-434c-95ec-93687ecd1396"
# left_vis = os.path.join(folder_path, "_iblrig_leftCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0.jpg")
# original_left_vis = os.path.join(folder_path, "_iblrig_leftCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_src_t17p0.jpg")

# # Build point cloud with background removed (builder-based workflow)
# pcd_builder_left = point_cloud.DepthMapImageToPointCloud(left_vis, input_format="bgr")
# # threshold = pcd_builder_right.interactive_pick_threshold()  # or interactive_depth_picker(right_vis)
# threshold = 0.15
# if threshold is not None:
#     pcd_builder_left.remove_background(threshold)
# pcd_builder_left.to_pcd()
# pcd_builder_left.show()
# # pcd_builder_left.save_ply(color_image_path=original_left_vis)