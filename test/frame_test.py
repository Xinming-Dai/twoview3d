import twoview3d.model.point_cloud as point_cloud
import os


folder_path = "./twoview3d/test/frames"
image_path = os.path.join(folder_path, "_iblrig_leftCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_vis_t17p0.jpg")
original_image_path = os.path.join(folder_path, "_iblrig_leftCamera.downsampled.ecb5520d-1358-434c-95ec-93687ecd1396_src_t17p0.jpg")

point_cloud.ImageToPointCloud(image_path, input_format="bgr").show_in_original_color(original_image_path)