# import math
# import numpy as np
# import pybullet as p
# import cv2
# import open3d as o3d
# import re
# from shapely.geometry import Point, Polygon,MultiPolygon
# from scipy.ndimage import binary_fill_holes
# import torch
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# from PIL import Image
# import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as R
# from refine.env.constants import WORKSPACE_LIMITS, PIXEL_SIZE
# from plyfile import PlyData, PlyElement
# from scipy.spatial import cKDTree
# from typing import Union

# from typing import Literal, Optional, Tuple, Union
# try:
#     from scipy.spatial import cKDTree
#     _HAS_SCIPY = True
# except Exception:
#     _HAS_SCIPY = False

# reconstruction_config = {
#     'nb_neighbors': 50,
#     'std_ratio': 2.0,
#     'voxel_size': 0.0015,
#     'icp_max_try': 5,
#     'icp_max_iter': 2000,
#     'translation_thresh': 3.95,
#     'rotation_thresh': 0.02,
#     'max_correspondence_distance': 0.02
# }

# graspnet_config = {
#     'graspnet_checkpoint_path': '/home/ubuntu/task/more_than_grasp/models/graspnet/checkpoints/checkpoint-rs.tar',
#     'refine_approach_dist': 0.01,
#     'dist_thresh': 0.05,
#     'angle_thresh': 10,
#     'mask_thresh': 0.5
# }
# # graspnet_config = {
# #     'graspnet_checkpoint_path': '/home/ubuntu/task/MyProject_Grasp-Push/models/graspnet/checkpoints/checkpoint-rs.tar',
# #     'refine_approach_dist': 0.015,
# #     'dist_thresh': 0.05,
# #     'angle_thresh': 4,
# #     'mask_thresh': 0.3
# # }
# def get_pointcloud(depth, intrinsics):
#     """Get 3D pointcloud from perspective depth image.
#     Args:
#         depth: HxW float array of perspective depth in meters.
#         intrinsics: 3x3 float array of camera intrinsics matrix.
#     Returns:
#         points: HxWx3 float array of 3D points in camera coordinates.
#     """
#     height, width = depth.shape
#     xlin = np.linspace(0, width - 1, width)
#     ylin = np.linspace(0, height - 1, height)
#     px, py = np.meshgrid(xlin, ylin)
#     px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
#     py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
#     points = np.float32([px, py, depth]).transpose(1, 2, 0)

#     return points

# def get_all_pointcloud(depth, intrinsics, seg, target_id):

#     height, width = depth.shape
#     xlin = np.linspace(0, width - 1, width)
#     ylin = np.linspace(0, height - 1, height)
#     px, py = np.meshgrid(xlin, ylin)
#     px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
#     py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
#     points = np.float32([px, py, depth]).transpose(1, 2, 0)

#     segm = np.array(seg, dtype=np.int32).reshape(depth.shape)
#     object_mask = (segm == target_id)|(segm == 1)
#     object_pcd_mask = (segm == target_id)

#     depth_mask = depth.copy()
#     depth_mask[~object_mask] = 0  

#     depth_pcd = depth.copy()
#     depth_pcd[~object_pcd_mask] = 0

#     # px, py = np.meshgrid(xlin, ylin)
#     # px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
#     # py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
#     points2 = np.float32([px, py, depth_mask]).transpose(1, 2, 0)
#     points_pcd = np.float32([px, py, depth_pcd]).transpose(1, 2, 0)

#     return points, points2, points_pcd

# def get_mask_pointcloud(depth, intrinsics, seg, target_id):
#     """

#     """
#     segm = np.array(seg, dtype=np.int32).reshape(depth.shape)
#     object_mask = (segm == target_id)|(segm == 1)
#     object_pcd_mask = (segm == target_id)
    
#     # segm = np.array(seg, dtype=np.int32).reshape(depth.shape)

#     # # segmentationMaskBuffer = objectUniqueId + (linkIndex+1)<<24
#     # obj_ids   = segm & ((1 << 24) - 1)    # 取低 24 bit => objectUniqueId
#     # # 根据 objectUniqueId 构造 mask（假设 target_id 是 bodyUniqueId）
#     # object_mask     = (obj_ids == target_id) | (obj_ids == 1)  # 1 可以是平面 / 桌子等
#     # object_pcd_mask = (obj_ids == target_id)

#     depth_mask = depth.copy()
#     depth_mask[~object_mask] = 0  

#     depth_pcd = depth.copy()
#     depth_pcd[~object_pcd_mask] = 0

#     height, width = depth.shape
#     xlin = np.linspace(0, width - 1, width)
#     ylin = np.linspace(0, height - 1, height)
#     px, py = np.meshgrid(xlin, ylin)
#     px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
#     py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
#     points = np.float32([px, py, depth_mask]).transpose(1, 2, 0)
#     points_pcd = np.float32([px, py, depth_pcd]).transpose(1, 2, 0)
#     return points, points_pcd

# def get_all_obj_mask_pointcloud(depth, intrinsics, seg, all_obj_id):
#     """
#     get all obj points for push sample
#     """
#     segm = np.array(seg, dtype=np.int32).reshape(depth.shape)
#     all_points = []

#     for id in all_obj_id:
#         object_pcd_mask = (segm == id)

#         depth_pcd = depth.copy()
#         depth_pcd[~object_pcd_mask] = 0

#         height, width = depth.shape
#         xlin = np.linspace(0, width - 1, width)
#         ylin = np.linspace(0, height - 1, height)
#         px, py = np.meshgrid(xlin, ylin)
#         px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
#         py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])

#         point_pcd = np.float32([px, py, depth_pcd]).transpose(1, 2, 0)
#         all_points.append(point_pcd)
#     return all_points

# def get_obj_pointcloud(depth, intrinsics, seg, target_id):
#     """

#     """
#     segm = np.array(seg, dtype=np.int32).reshape(depth.shape)
#     object_mask = (segm == target_id)
    
#     depth = depth.copy()
#     depth[~object_mask] = 0  
    
#     height, width = depth.shape
#     xlin = np.linspace(0, width - 1, width)
#     ylin = np.linspace(0, height - 1, height)
#     px, py = np.meshgrid(xlin, ylin)
#     px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
#     py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
#     points = np.float32([px, py, depth]).transpose(1, 2, 0)

#     return points 

# def transform_pointcloud(points, transform):
#     """Apply rigid transformation to 3D pointcloud.
#     Args:
#         points: HxWx3 float array of 3D points in camera coordinates.
#         transform: 4x4 float array representing a rigid transformation matrix.
#     Returns:
#         points: HxWx3 float array of transformed 3D points.
#     """
#     padding = ((0, 0), (0, 0), (0, 1))
#     homogen_points = np.pad(points.copy(), padding, "constant", constant_values=1)
#     for i in range(3):
#         points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
#     return points

# def process_pcds(pcds, reconstruction_config):
#     trans = dict()
#     pcd = pcds[0]
#     pcd.estimate_normals()
#     pcd, _ = pcd.remove_statistical_outlier(
#         nb_neighbors = reconstruction_config['nb_neighbors'],
#         std_ratio = reconstruction_config['std_ratio']
#     )
#     for i in range(1, len(pcds)):
#         voxel_size = reconstruction_config['voxel_size']
#         income_pcd, _ = pcds[i].remove_statistical_outlier(
#             nb_neighbors = reconstruction_config['nb_neighbors'],
#             std_ratio = reconstruction_config['std_ratio']
#         )
#         income_pcd.estimate_normals()
#         income_pcd = income_pcd.voxel_down_sample(voxel_size)
#         transok_flag = False
#         for _ in range(reconstruction_config['icp_max_try']): # try 5 times max
#             reg_p2p = o3d.pipelines.registration.registration_icp(
#                 income_pcd,
#                 pcd,
#                 reconstruction_config['max_correspondence_distance'],
#                 np.eye(4, dtype = np.float),
#                 o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#                 o3d.pipelines.registration.ICPConvergenceCriteria(reconstruction_config['icp_max_iter'])
#             )
#             if (np.trace(reg_p2p.transformation) > reconstruction_config['translation_thresh']) \
#                 and (np.linalg.norm(reg_p2p.transformation[:3, 3]) < reconstruction_config['rotation_thresh']):
#                 # trace for transformation matrix should be larger than 3.5
#                 # translation should less than 0.05
#                 transok_flag = True
#                 break
#         if not transok_flag:
#             reg_p2p.transformation = np.eye(4, dtype = np.float32)
#         income_pcd = income_pcd.transform(reg_p2p.transformation)
#         trans[i] = reg_p2p.transformation
#         pcd += income_pcd
#         pcd = pcd.voxel_down_sample(voxel_size)
#         pcd.estimate_normals()
#     return trans, pcd

# def process_pcds_test(pcds):
#     points_state_list = []
#     colors = []
#     for pcd in pcds:
#         points = np.asarray(pcd.points)
#         color = np.asarray(pcd.colors)
#         points_state_list.append(points)
#         colors.append(color)

#     points_state = np.vstack(points_state_list)
#     colors_state = np.vstack(colors)
#     points_pcd = o3d.geometry.PointCloud()
#     points_pcd.points = o3d.utility.Vector3dVector(points_state)
#     points_pcd.colors = o3d.utility.Vector3dVector(colors_state)

#     return points_pcd

# def process_all_pcds(all_config_pcd):
#     obj_points_list = []

#     for i in range(len(all_config_pcd[0])):
#         obj_point = []
#         for j in range(len(all_config_pcd)):
#             pcd = all_config_pcd[j][i]
#             points = np.asarray(pcd.points)

#             obj_point.append(points)

#         obj_point = np.vstack(obj_point)
#         obj_points_list.append(obj_point)
#     obj_pcds_list = []
#     for i in range(len(obj_points_list)):
#         points_pcd = o3d.geometry.PointCloud()
#         points_pcd.points = o3d.utility.Vector3dVector(obj_points_list[i])
#         obj_pcds_list.append(points_pcd)

#     return obj_pcds_list

# def get_fuse_pointcloud(env, obj_id, id=0):
#     pcds = []
#     configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
#     # Capture near-orthographic RGB-D images and segmentation masks.
#     for config in configs:
#         color, depth, seg = env.render_camera(config)
#         if id == 0:
#             xyz, _ = get_mask_pointcloud(depth, config["intrinsics"], seg, obj_id)
#         else:
#             _, xyz = get_mask_pointcloud(depth, config["intrinsics"], seg, obj_id)
#         # xyz = get_pointcloud(depth, config["intrinsics"])
#         position = np.array(config["position"]).reshape(3, 1)
#         rotation = p.getMatrixFromQuaternion(config["rotation"])
#         rotation = np.array(rotation).reshape(3, 3)
#         transform = np.eye(4)
#         transform[:3, :] = np.hstack((rotation, position))
#         points = transform_pointcloud(xyz, transform)
#         # Filter out 3D points that are outside of the predefined bounds.
#         ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
#         iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
#         iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
#         valid = ix & iy & iz
#         points = points[valid]
#         colors = color[valid]
#         # Sort 3D points by z-value, which works with array assignment to simulate
#         # z-buffering for rendering the heightmap image.
#         iz = np.argsort(points[:, -1])
#         points, colors = points[iz], colors[iz]

#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
#         # pcd.voxel_down_sample(reconstruction_config['voxel_size'])
#         # # visualization
#         # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#         # o3d.visualization.draw_geometries([pcd, frame])
#         # the first pcd is the one for start fusion
#         pcds.append(pcd)

#     # _, fuse_pcd = process_pcds(pcds, reconstruction_config)
#     fuse_pcd = process_pcds_test(pcds)
#     # visualization
#     # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#     # o3d.visualization.draw_geometries([fuse_pcd, frame])

#     return fuse_pcd

# def get_all_obj_pointcloud(env, obj_lis):
#     all_config_pcds = []
#     configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
#     # Capture near-orthographic RGB-D images and segmentation masks.
#     for config in configs:
#         one_config_pcds = []
#         color, depth, seg = env.render_camera(config)
#         all_xyz = get_all_obj_mask_pointcloud(depth, config['intrinsics'], seg, obj_lis)
#         # xyz = get_pointcloud(depth, config["intrinsics"])
#         for xyz in all_xyz:
#             position = np.array(config["position"]).reshape(3, 1)
#             rotation = p.getMatrixFromQuaternion(config["rotation"])
#             rotation = np.array(rotation).reshape(3, 3)
#             transform = np.eye(4)
#             transform[:3, :] = np.hstack((rotation, position))
#             points = transform_pointcloud(xyz, transform)
#             # Filter out 3D points that are outside of the predefined bounds.
#             ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
#             iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
#             iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
#             valid = ix & iy & iz
#             points = points[valid]
#             colors = color[valid]
#             # Sort 3D points by z-value, which works with array assignment to simulate
#             # z-buffering for rendering the heightmap image.
#             iz = np.argsort(points[:, -1])
#             points, colors = points[iz], colors[iz]

#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(points)
#             pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
#             pcd.voxel_down_sample(reconstruction_config['voxel_size'])
#             # # visualization
#             # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#             # o3d.visualization.draw_geometries([pcd, frame])
#             # the first pcd is the one for start fusion
#             one_config_pcds.append(pcd)
#         all_config_pcds.append(one_config_pcds)
#     # _, fuse_pcd = process_pcds(pcds, reconstruction_config)
#     fuse_pcd = process_all_pcds(all_config_pcds)
#     # visualization
#     # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#     # o3d.visualization.draw_geometries([fuse_pcd, frame])

#     return fuse_pcd


# def global_label_points(depth, intrinsics, seg, target_id):
#     """
#     assign labels to goal-obj and other obj
#     goal-obj:[0,1,0] other:[0,0,1]
#     """
#     segm = np.array(seg, dtype=np.int32).reshape(depth.shape)
#     goal_obj_mask = (segm == target_id)
#     other_mask = (segm != target_id) & (segm != 1)  
#     without_floor = (segm == 1)
#     # crop floor
#     depth_mask_global = depth.copy()
#     depth_mask_global[without_floor] = 0
#     # depth_mask_obj = depth.copy()
#     # depth_mask_obj[~goal_obj_mask] = 0
#     height, width = depth.shape
#     xlin = np.linspace(0, width - 1, width)
#     ylin = np.linspace(0, height - 1, height)
#     px, py = np.meshgrid(xlin, ylin)
#     px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
#     py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    
#     points = np.float32([px, py, depth_mask_global]).transpose(1, 2, 0)
#     # obj_points = np.float32([px, py, depth_mask_obj]).transpose(1, 2, 0)
#     # add labels to global_pc
#     labels = np.zeros((height, width, 2), dtype=np.float32)
#     labels[goal_obj_mask] = [1.0, 0.0]  # goal-obj
#     labels[other_mask] = [0.0, 1.0] # other-obj
#     global_points_six = np.concatenate([points, labels], axis=-1) 

#     return global_points_six

# def get_global_pc(env):
#     pcds = []
#     segs = []
#     configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
#     # Capture near-orthographic RGB-D images and segmentation masks.
#     for config in configs:
#         color, depth, seg = env.render_camera(config)
#         xyz = get_pointcloud(depth, config["intrinsics"])
        
#         # xyz = get_pointcloud(depth, config["intrinsics"])
#         position = np.array(config["position"]).reshape(3, 1)
#         rotation = p.getMatrixFromQuaternion(config["rotation"])
#         rotation = np.array(rotation).reshape(3, 3)
#         transform = np.eye(4)
#         transform[:3, :] = np.hstack((rotation, position))
#         # transform pc from camera_base to robot_base
#         points = transform_pointcloud(xyz, transform)
#         # Filter out 3D points that are outside of the predefined bounds.
#         ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
#         iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
#         iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
#         valid = ix & iy & iz
#         points = points[valid]
#         colors = color[valid]
#         # Sort 3D points by z-value, which works with array assignment to simulate
#         # z-buffering for rendering the heightmap image.
#         iz = np.argsort(points[:, -1])
#         points, colors = points[iz], colors[iz]

#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
#         pcd.voxel_down_sample(reconstruction_config['voxel_size'])
#         # # visualization
#         # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#         # o3d.visualization.draw_geometries([pcd, frame])
#         # the first pcd is the one for start fusion
#         pcds.append(pcd)
#         segs.append(seg)

#     # _, fuse_pcd = process_pcds(pcds, reconstruction_config)
#     fuse_pcd = process_pcds_test(pcds)
#     # ply_global = furthest_point_sampling(fuse_pcd, n_samples=25000)
#     # ply_global_for_eval = furthest_point_sampling(fuse_pcd, n_samples=18000)
#     # pcd_global = o3d.geometry.PointCloud()
#     # pcd_global.points = o3d.utility.Vector3dVector(ply_global)
#     # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#     # o3d.visualization.draw_geometries([pcd_global, frame])
#     return fuse_pcd, segs[0]

# def get_global_label_pc(env, target_id):
#     pcds = []
#     segs = []
#     configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
#     # Capture near-orthographic RGB-D images and segmentation masks.
#     for config in configs:
#         # 渲染
#         color, depth, seg = env.render_camera(config)

#         # HxWx5: [x,y,z, l0, l1]，其中 l 为 one-hot（例如 [0,1] 代表目标，[0,0,1] 则是你三类时的第三列，按你实际为两类或三类来）
#         xyz_label = global_label_points(depth, config["intrinsics"], seg, target_id)  # HxWx5

#         # 拆分
#         xyz_hw   = xyz_label[:, :, :3].astype(np.float64)   # HxWx3
#         labels_hw = xyz_label[:, :, 3:].astype(np.float32)  # HxWx2
#         H, W = depth.shape

#         # 位姿变换（保持 HxWx3，再展平）
#         position = np.array(config["position"]).reshape(3, 1)
#         rotation = np.array(p.getMatrixFromQuaternion(config["rotation"])).reshape(3, 3)
#         T = np.eye(4); T[:3, :3] = rotation; T[:3, 3] = position[:, 0]

#         points_hw = transform_pointcloud(xyz_hw, T)         # HxWx3
#         points    = points_hw.reshape(-1, 3)                # N x 3
#         labels    = labels_hw.reshape(-1, labels_hw.shape[-1])  # N x C (C=2或3)
#         colors    = color.reshape(-1, 3).astype(np.float64)      # N x 3

#         # 工作空间过滤（labels/colors 同步）
#         ix = (points[:, 0] >= env.bounds[0, 0]) & (points[:, 0] < env.bounds[0, 1])
#         iy = (points[:, 1] >= env.bounds[1, 0]) & (points[:, 1] < env.bounds[1, 1])
#         iz = (points[:, 2] >= env.bounds[2, 0]) & (points[:, 2] < env.bounds[2, 1])
#         valid = ix & iy & iz

#         points = points[valid]
#         labels = labels[valid]
#         colors = colors[valid]

#         # 按 z 排序（明确用 [:,2]）
#         order = np.argsort(points[:, 2])
#         points = points[order]
#         labels = labels[order]
#         colors = colors[order]

#         # Open3D 点云（仅几何/颜色）
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

#         # 体素降采样 + trace（拿原始索引）
#         voxel = reconstruction_config['voxel_size']
#         min_b = pcd.get_min_bound() - voxel
#         max_b = pcd.get_max_bound() + voxel

#         pcd_ds, _, traces = pcd.voxel_down_sample_and_trace(
#             voxel_size=voxel,
#             min_bound=min_b,
#             max_bound=max_b,
#             approximate_class=True
#         )

#         xyz_ds = np.asarray(pcd_ds.points).astype(np.float32)   # (M,3)
#         # 继承标签：体素内第一个原始点的标签
#         keep_idx = np.array([np.asarray(idx, dtype=np.int64)[0] for idx in traces], dtype=np.int64)
#         # 若想用“距离体素输出点最近”的原始点标签：
#         # keep_idx = np.array([ idxs[np.argmin(np.linalg.norm(points[np.asarray(idxs)] - xyz_ds[i], axis=1))]
#         #                       for i, idxs in enumerate(traces) ], dtype=np.int64)
#         # 若想用“多数投票”（one-hot 求和后 argmax）：
#         # lab_ds = np.stack([labels[np.asarray(idxs)].sum(0) for idxs in traces], axis=0)
#         # lab_ds = (lab_ds == lab_ds.max(axis=1, keepdims=True)).astype(np.float32)

#         lab_ds = labels[keep_idx].astype(np.float32)           # (M,C)

#         # 重新拼装 (M, 3+C)  -> 你的 C=2 时就是 (M,5)
#         xyz_label_ds = np.concatenate([xyz_ds, lab_ds], axis=1).astype(np.float32)

#         # # visualization
#         # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#         # o3d.visualization.draw_geometries([pcd, frame])
#         # the first pcd is the one for start fusion
#         pcds.append(xyz_label_ds)
#         segs.append(seg)

#     # _, fuse_pcd = process_pcds(pcds, reconstruction_config)
#     # fuse_pcd = process_pcds_test(pcds)
#     points_state = np.vstack(pcds)

#     ply_global = fps_xyz_label(points_state, n_samples=25000)
#     # ply_global_for_eval = furthest_point_sampling(fuse_pcd, n_samples=18000)
#     # pcd_global = o3d.geometry.PointCloud()
#     # pcd_global.points = o3d.utility.Vector3dVector(ply_global)
#     # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#     # o3d.visualization.draw_geometries([pcd_global, frame])
#     return ply_global, segs[0]

# def get_pcd_for_all(env, obj_id):
#     pcd_g = []
#     pcd_o = []
#     pcd_for_graspnet = []
#     configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
#     # Capture near-orthographic RGB-D images and segmentation masks.
#     for config in configs:
#         color, depth, seg = env.render_camera(config)
#         xyz1, xyz2, xyz3 = get_all_pointcloud(depth, config["intrinsics"], seg, obj_id)
#         # xyz = get_pointcloud(depth, config["intrinsics"])
#         position = np.array(config["position"]).reshape(3, 1)
#         rotation = p.getMatrixFromQuaternion(config["rotation"])
#         rotation = np.array(rotation).reshape(3, 3)
#         transform = np.eye(4)
#         transform[:3, :] = np.hstack((rotation, position))
#         all_xyz = [xyz1, xyz2, xyz3]
#         i = 0
#         for xyz in all_xyz:
#             points = transform_pointcloud(xyz, transform)
#             # Filter out 3D points that are outside of the predefined bounds.

#             ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
#             iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
#             iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
#             valid = ix & iy & iz
#             points = points[valid]
#             colors = color[valid]
#             # Sort 3D points by z-value, which works with array assignment to simulate
#             # z-buffering for rendering the heightmap image.
#             iz = np.argsort(points[:, -1])
#             points, colors = points[iz], colors[iz]

#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(points)
#             pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
#             pcd.voxel_down_sample(reconstruction_config['voxel_size'])
#             # # visualization
#             # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#             # o3d.visualization.draw_geometries([pcd, frame])
#             # the first pcd is the one for start fusion
#             if(i == 0):
#                 pcd_g.append(pcd)
#                 i += 1
#             elif(i == 1):
#                 pcd_for_graspnet.append(pcd)
#                 i += 1
#             else:
#                 pcd_o.append(pcd)

#     _, fuse_pcd1 = process_pcds(pcd_g, reconstruction_config)
#     _, fuse_pcd2 = process_pcds(pcd_for_graspnet, reconstruction_config)
#     _, fuse_pcd3 = process_pcds(pcd_o, reconstruction_config)
#     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#     o3d.visualization.draw_geometries([fuse_pcd1, frame])
#     o3d.visualization.draw_geometries([fuse_pcd2, frame])
#     o3d.visualization.draw_geometries([fuse_pcd3, frame])
#     return fuse_pcd1, fuse_pcd2, fuse_pcd3

# def get_global_pc_from_multi_view(env):
#     """
#     set two fixed cameras with 45-degree tilt angle to get dense pc.
#     """
#     pcds = []
#     segs = []
#     configs = [env.agent_cams[1], env.agent_cams[2]]
    
#     # Capture near-orthographic RGB-D images and segmentation masks.
#     for config in configs:
#         color, depth, seg = env.render_camera(config)
#         xyz = get_pointcloud(depth, config["intrinsics"])
#         # xyz = get_pointcloud(depth, config["intrinsics"])
#         position = np.array(config["position"]).reshape(3, 1)
#         rotation = p.getMatrixFromQuaternion(config["rotation"])
#         rotation = np.array(rotation).reshape(3, 3)
#         transform = np.eye(4)
#         transform[:3, :] = np.hstack((rotation, position))
#         # transform pc from camera_base to robot_base
#         points = transform_pointcloud(xyz, transform)
#         # Filter out 3D points that are outside of the predefined bounds.
#         ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
#         iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
#         iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
#         valid = ix & iy & iz

#         valid = np.isfinite(depth) & (depth > 0)            
#         valid &= np.isfinite(points).all(axis=-1) 

#         points = points[valid]
#         colors = color[valid]
#         # Sort 3D points by z-value, which works with array assignment to simulate
#         # z-buffering for rendering the heightmap image.
#         iz = np.argsort(points[:, -1])
#         points, colors = points[iz], colors[iz]

#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
#         # pcd.voxel_down_sample(reconstruction_config['voxel_size'])
#         # # visualization
#         # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#         # o3d.visualization.draw_geometries([pcd, frame])
#         # the first pcd is the one for start fusion
#         pcds.append(pcd)
#         segs.append(seg)

#     fuse_pcd = process_pcds_test(pcds)
#     # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#     # o3d.visualization.draw_geometries([pcd_global, frame])
#     return fuse_pcd, pcd

# def get_obj_pc_from_multi_view(env, obj_id):
#     """
#     set two fixed cameras with 45-degree tilt angle to get dense pc.
#     """
#     np.random.seed(1239)
#     pcds = []
#     configs = [env.agent_cams[1], env.agent_cams[2]]
#     # Capture near-orthographic RGB-D images and segmentation masks.
#     for config in configs:
#         color, depth, seg = env.render_camera(config)
#         _, xyz = get_mask_pointcloud(depth, config["intrinsics"], seg, obj_id)
#         position = np.array(config["position"]).reshape(3, 1)
#         rotation = p.getMatrixFromQuaternion(config["rotation"])
#         rotation = np.array(rotation).reshape(3, 3)
#         transform = np.eye(4)
#         transform[:3, :] = np.hstack((rotation, position))
#         points = transform_pointcloud(xyz, transform)
#         # Filter out 3D points that are outside of the predefined bounds.
#         ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
#         iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
#         iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
#         valid = ix & iy & iz

#         # no filter
#         valid = (seg == obj_id)
#         valid &= np.isfinite(depth) & (depth > 0)
#         valid &= np.isfinite(points).all(axis=-1)
#         valid &= (np.linalg.norm(points, axis=-1) > 1e-9)

#         points = points[valid]
#         colors = color[valid]
#         # Sort 3D points by z-value, which works with array assignment to simulate
#         # z-buffering for rendering the heightmap image.
#         iz = np.lexsort((points[:,1], points[:,0], points[:,2]))
#         points, colors = points[iz], colors[iz]

#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
#         pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
#         # pcd.voxel_down_sample(reconstruction_config['voxel_size'])
#         # # visualization
#         # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#         # o3d.visualization.draw_geometries([pcd, frame])
#         # the first pcd is the one for start fusion
#         pcds.append(pcd)

#     # _, fuse_pcd = process_pcds(pcds, reconstruction_config)
#     fuse_pcd = process_pcds_test(pcds)
#     # visualization
#     # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#     # o3d.visualization.draw_geometries([fuse_pcd, frame])

#     return fuse_pcd, pcd

# def adjust_pose_z_axis_to_down(rot_matrix):

#     # 保留原始 X 轴方向（可选）
#     x_axis = rot_matrix[:, 0]  # shape (3,)

#     # 目标 Z 轴为竖直向下
#     z_axis = np.array([0, 0, -1])

#     # 重新计算 Y，使得 XYZ 成为右手坐标系
#     y_axis = np.cross(z_axis, x_axis)
#     y_axis /= np.linalg.norm(y_axis)

#     # 重新计算正交的 X
#     x_axis = np.cross(y_axis, z_axis)
#     x_axis /= np.linalg.norm(x_axis)

#     # 构造新旋转矩阵
#     new_rot = np.stack([x_axis, y_axis, z_axis], axis=1)  # 每列是一个轴

#     # 转为四元数
#     new_quat = R.from_matrix(new_rot).as_quat()  # [x, y, z, w]
#     return new_quat, new_rot

# def furthest_point_sampling(points, colors=None, semantics=None, n_samples=4096):
#     """
#     points: [N, 3] tensor containing the whole point cloud
#     n_samples: samples you want in the sampled point cloud typically &lt;&lt; N
#     """
#     # Convert points to PyTorch tensor if not already and move to GPU
#     # print(colors)
#     pcd_np = np.asarray(points.points)
#     # pcd_np = points.cpu().numpy()
#     points = torch.from_numpy(pcd_np).float().cuda()  # [N, 3]
#     # points = points.to('cuda')
#     if colors is not None:
#         colors = torch.Tensor(colors).cuda()
#     if semantics is not None:
#         semantics = semantics.astype(np.int32)
#         semantics = torch.Tensor(semantics).cuda()

#     # Number of points
#     num_points = points.size(0)  # N

#     # Initialize an array for the sampled indices
#     sample_inds = torch.zeros(n_samples, dtype=torch.long).cuda()  # [S]

#     # Initialize distances to inf
#     dists = torch.ones(num_points).cuda() * float("inf")  # [N]

#     # Select the first point randomly
#     selected = torch.randint(num_points, (1,), dtype=torch.long).cuda()  # [1]
#     sample_inds[0] = selected

#     # Iteratively select points for a maximum of n_samples
#     for i in range(1, n_samples):
#         # Find the distance to the last added point in selected
#         last_added = sample_inds[i - 1]  # Scalar
#         dist_to_last_added_point = torch.sum(
#             (points[last_added] - points) ** 2, dim=-1
#         )  # [N]

#         # If closer, update distances
#         dists = torch.min(dist_to_last_added_point, dists)  # [N]

#         # Pick the one that has the largest distance to its nearest neighbor in the sampled set
#         selected = torch.argmax(dists)  # Scalar
#         sample_inds[i] = selected

#     if colors is not None and semantics is not None:
#         return (
#             points[sample_inds].cpu().numpy(),
#             colors[sample_inds].cpu().numpy(),
#             semantics[sample_inds].cpu().numpy(),
#         )  # [S, 3]
#     elif colors is not None:
#         return points[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy()
#     else:
#         # pcd = o3d.geometry.PointCloud()
#         # pcd.points = o3d.utility.Vector3dVector(points[sample_inds].cpu().numpy())
#         # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#         # o3d.visualization.draw_geometries([pcd, frame])
#         return points[sample_inds]

# def fps_xyz_label(xyz_label: np.ndarray,
#                   n_samples: int = 4096,
#                   return_index: bool = False,
#                   device: str = None,
#                   start_idx: int = None,
#                   seed: int = None):
#     """
#     Furthest Point Sampling on xyz and gather corresponding one-hot labels.

#     Args:
#         xyz_label: (N, 3+C) numpy array, first 3 are xyz, remaining C are labels (one-hot).
#                    你的场景: C=2 -> (N,5)
#         n_samples: 输出点数 S，自动截断到 N 以内
#         return_index: 是否额外返回采样到的原始索引 (S,)
#         device: 'cuda' 或 'cpu'；默认自动选择（DataLoader worker 内建议显式传 'cpu'）
#         start_idx: 首个点的索引；为 None 时随机选择
#         seed: 若需要确定性随机起点，给一个种子

#     Returns:
#         sampled: (S, 3+C) numpy array
#         (可选) idx: (S,) numpy int64 indices
#     """
#     arr = np.asarray(xyz_label)
#     assert arr.ndim == 2 and arr.shape[1] >= 4, "xyz_label 应为 (N, 3+C)"

#     N, D = arr.shape
#     C = D - 3
#     S = min(n_samples, N)

#     # 设备选择
#     # if device is None:
#     #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # DataLoader worker 内如未使用 spawn，建议强制 cpu
#     device = 'cuda'

#     xyz = torch.from_numpy(arr[:, :3]).to(device=device, dtype=torch.float32)   # [N,3]
#     lab = torch.from_numpy(arr[:, 3:]).to(device=device, dtype=torch.float32)   # [N,C]

#     # 采样索引与距离
#     idx = torch.empty(S, dtype=torch.long, device=device)
#     dists = torch.full((N,), float('inf'), device=device)

#     # 首点
#     if start_idx is None:
#         if seed is not None:
#             g = torch.Generator(device=device)
#             g.manual_seed(int(seed))
#             idx0 = torch.randint(N, (1,), generator=g, device=device)[0]
#         else:
#             idx0 = torch.randint(N, (1,), device=device)[0]
#     else:
#         idx0 = torch.tensor(start_idx, dtype=torch.long, device=device).clamp_(0, N-1)

#     idx[0] = idx0

#     # 迭代选点（O(N*S)）
#     for i in range(1, S):
#         last = idx[i-1]
#         dist2 = torch.sum((xyz - xyz[last])**2, dim=-1)  # [N]
#         dists = torch.minimum(dists, dist2)
#         idx[i] = torch.argmax(dists)

#     # 收集结果
#     sampled_xyz = xyz[idx]          # (S,3)
#     sampled_lab = lab[idx]          # (S,C)
#     sampled = torch.cat([sampled_xyz, sampled_lab], dim=1).cpu().numpy()  # (S, 3+C)

#     if return_index:
#         return sampled, idx.cpu().numpy().astype(np.int64)
#     else:
#         return sampled

# def furthest_point_sampling_nocuda(points, colors=None, semantics=None, n_samples=4096,start_idx=0):
#     """
#     points: [N, 3] tensor containing the whole point cloud
#     n_samples: samples you want in the sampled point cloud typically &lt;&lt; N
#     """

#     # Number of points
#     num_points = points.shape[0] # N

#     # Initialize an array for the sampled indices
#     sample_inds = torch.zeros(n_samples, dtype=torch.long, device=points.device) # [S]

#     # Initialize distances to inf
#     dists = torch.ones(num_points) * float("inf")  # [N]

#     # Select the first point randomly
#     # selected = torch.randint(num_points, (1,), dtype=torch.long)  # [1]
#     selected = torch.tensor([start_idx], dtype=torch.long, device=points.device)
#     sample_inds[0] = selected

#     # Iteratively select points for a maximum of n_samples
#     for i in range(1, n_samples):
#         # Find the distance to the last added point in selected
#         last_added = sample_inds[i - 1]  # Scalar
#         dist_to_last_added_point = torch.sum(
#             (points[last_added] - points) ** 2, dim=-1
#         )  # [N]

#         # If closer, update distances
#         dists = torch.min(dist_to_last_added_point, dists)  # [N]

#         # Pick the one that has the largest distance to its nearest neighbor in the sampled set
#         selected = torch.argmax(dists)  # Scalar
#         sample_inds[i] = selected

#         # pcd = o3d.geometry.PointCloud()
#         # pcd.points = o3d.utility.Vector3dVector(points[sample_inds].cpu().numpy())
#         # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#         # o3d.visualization.draw_geometries([pcd, frame])
#     return points[sample_inds], sample_inds



# # def fps_p3d(points: torch.Tensor,
# #                   n_samples: int,
# #                   start_idx: int = 0,
# #                   return_index: bool = True):
# #     """
# #     points: [N, C>=3] torch.Tensor
# #     返回: out_points [n_samples, C]（始终固定长度）
# #     """
# #     from pytorch3d.ops import sample_farthest_points

# #     assert isinstance(points, torch.Tensor), "points 必须是 torch.Tensor"
# #     assert points.ndim == 2 and points.shape[1] >= 3, f"points 需为 [N,C>=3]，但得到 {points.shape}"

# #     device = points.device
# #     N = int(points.shape[0])
# #     S = int(n_samples)

# #     if N == 0:
# #         raise ValueError("points 为空，无法 FPS")

# #     K = min(S, N)

# #     xyz = points[:, :3].to(dtype=torch.float32).contiguous().unsqueeze(0)  # [1,N,3]

# #     # 固定起点：交换到第 0 个，再 random_start_point=False
# #     if start_idx is not None:
# #         si = int(start_idx)
# #         if not (0 <= si < N):
# #             raise ValueError(f"start_idx 超界: {start_idx} (总点数 {N})")

# #         perm = torch.arange(N, device=device)
# #         if si != 0:
# #             perm[0], perm[si] = perm[si], perm[0]
# #         xyz_perm = xyz[:, perm, :]
# #         _, idx_perm = sample_farthest_points(xyz_perm, K=K, random_start_point=False)
# #         sel = perm[idx_perm[0]]  # [K]
# #     else:
# #         _, idx = sample_farthest_points(xyz, K=K, random_start_point=False)
# #         sel = idx[0]  # [K]

# #     if K < S:
# #         pad = sel.new_full((S - K,), sel[0].item())
# #         sel = torch.cat([sel, pad], dim=0)

# #     out = points[sel]  # [S, C]
# #     return (out, sel) if return_index else out

# def fps_p3d(
#     points: torch.Tensor,
#     n_samples: int,
#     start_idx: int = None,
#     return_index: bool = True,
# ):
#     """
#     points: [N, C>=3] torch.Tensor
#     返回: out_points [S, C]（固定长度）
#     """
#     from pytorch3d.ops import sample_farthest_points

#     assert isinstance(points, torch.Tensor), "points 必须是 torch.Tensor"
#     assert points.ndim == 2 and points.shape[1] >= 3, f"points 需为 [N,C>=3]，但得到 {points.shape}"

#     device = points.device
#     N = int(points.shape[0])
#     S = int(n_samples)

#     if N == 0:
#         raise ValueError("points 为空，无法 FPS")

#     # 1) 当 N <= S 时，没必要做 FPS：直接保留全部点，再 pad
#     if N <= S:
#         sel = torch.arange(N, device=device, dtype=torch.long)
#         if N < S:
#             pad = sel[:1].expand(S - N)   # 避免 item() 同步
#             sel = torch.cat([sel, pad], dim=0)
#         out = points[sel]
#         return (out, sel) if return_index else out

#     # 2) 只在需要时转换 dtype / contiguous
#     xyz = points[:, :3]
#     if xyz.dtype != torch.float32:
#         xyz = xyz.float()
#     if not xyz.is_contiguous():
#         xyz = xyz.contiguous()
#     xyz = xyz.unsqueeze(0)  # [1, N, 3]

#     # 3) 默认不重排；只有确实指定非 0 起点时才重排
#     if start_idx is None or int(start_idx) == 0:
#         _, idx = sample_farthest_points(xyz, K=S, random_start_point=False)
#         sel = idx[0]
#     else:
#         si = int(start_idx)
#         if not (0 <= si < N):
#             raise ValueError(f"start_idx 超界: {start_idx} (总点数 {N})")

#         perm = torch.arange(N, device=device)
#         perm0 = perm[0].clone()
#         permi = perm[si].clone()
#         perm[0] = permi
#         perm[si] = perm0

#         xyz_perm = xyz[:, perm, :]
#         _, idx_perm = sample_farthest_points(xyz_perm, K=S, random_start_point=False)
#         sel = perm[idx_perm[0]]

#     out = points[sel]
#     return (out, sel) if return_index else out

# def pad_pointcloud_list(points_list, pad_value: float = 0.0):
#     """
#     points_list: list of [Ni, C] torch.Tensor, Ni 可变, C 相同
#     返回:
#         padded : [B, Nmax, C]
#         lengths: [B]
#     """
#     assert isinstance(points_list, (list, tuple)) and len(points_list) > 0, "points_list 不能为空"

#     first = points_list[0]
#     assert isinstance(first, torch.Tensor) and first.ndim == 2, "每个元素都必须是 [N,C] 的 torch.Tensor"

#     device = first.device
#     dtype = first.dtype
#     C = first.shape[1]
#     B = len(points_list)

#     lengths = torch.empty((B,), dtype=torch.long, device=device)
#     max_n = 0

#     for i, p in enumerate(points_list):
#         assert isinstance(p, torch.Tensor), f"第 {i} 个元素不是 torch.Tensor"
#         assert p.ndim == 2, f"第 {i} 个元素维度错误: {p.shape}"
#         assert p.shape[1] == C, f"第 {i} 个元素通道数不一致: {p.shape[1]} vs {C}"
#         assert p.device == device, f"第 {i} 个元素 device 不一致"
#         lengths[i] = p.shape[0]
#         if p.shape[0] > max_n:
#             max_n = p.shape[0]

#     if max_n <= 0:
#         raise ValueError("所有点云都为空，无法打包")

#     padded = torch.full((B, max_n, C), pad_value, dtype=dtype, device=device)
#     for i, p in enumerate(points_list):
#         n = p.shape[0]
#         if n > 0:
#             padded[i, :n] = p

#     return padded, lengths

# def fps_p3d_batch_padded(
#     points_padded: torch.Tensor,
#     lengths: torch.Tensor,
#     n_samples: int,
#     start_idx=None,
#     return_index: bool = True,
# ):
#     """
#     points_padded: [B, Nmax, C>=3]
#     lengths      : [B]
#     返回:
#         out : [B, S, C]
#         idx : [B, S]
#     """
#     from pytorch3d.ops import sample_farthest_points

#     assert isinstance(points_padded, torch.Tensor), "points_padded 必须是 torch.Tensor"
#     assert points_padded.ndim == 3 and points_padded.shape[2] >= 3, \
#         f"points_padded 需为 [B,N,C>=3]，但得到 {points_padded.shape}"
#     assert isinstance(lengths, torch.Tensor), "lengths 必须是 torch.Tensor"
#     assert lengths.ndim == 1 and lengths.shape[0] == points_padded.shape[0], \
#         f"lengths 需为 [B]，但得到 {lengths.shape}"

#     device = points_padded.device
#     B, Nmax, C = points_padded.shape
#     S = int(n_samples)

#     lengths = lengths.to(device=device, dtype=torch.long)
#     if torch.any(lengths <= 0):
#         raise ValueError("lengths 中存在 <= 0 的项，无法 FPS")

#     # start_idx 当前先支持 None / 0 / 全 0；
#     # 若你后面确实要每个 batch 用不同起点，再单独扩展。
#     if start_idx is not None:
#         if isinstance(start_idx, int):
#             if start_idx != 0:
#                 raise NotImplementedError("batched FPS 暂不支持统一非 0 start_idx")
#         elif isinstance(start_idx, torch.Tensor):
#             if torch.any(start_idx.to(device=device, dtype=torch.long) != 0):
#                 raise NotImplementedError("batched FPS 暂不支持逐 batch 非 0 start_idx")
#         else:
#             raise TypeError("start_idx 只能是 None / int / torch.Tensor")

#     xyz = points_padded[:, :, :3]
#     if xyz.dtype != torch.float32:
#         xyz = xyz.float()
#     if not xyz.is_contiguous():
#         xyz = xyz.contiguous()

#     idx_out = torch.empty((B, S), dtype=torch.long, device=device)

#     # 1) 小点云：完全保持你原来的单点逻辑
#     small_mask = lengths <= S
#     if torch.any(small_mask):
#         small_ids = torch.nonzero(small_mask, as_tuple=False).flatten()
#         for b in small_ids.tolist():
#             n = int(lengths[b].item())
#             sel = torch.arange(n, device=device, dtype=torch.long)
#             if n < S:
#                 pad = sel[:1].expand(S - n)
#                 sel = torch.cat([sel, pad], dim=0)
#             idx_out[b] = sel

#     # 2) 大点云：真正 batched FPS
#     large_mask = lengths > S
#     if torch.any(large_mask):
#         large_ids = torch.nonzero(large_mask, as_tuple=False).flatten()
#         xyz_large = xyz[large_ids]              # [Bl, Nmax, 3]
#         lengths_large = lengths[large_ids]      # [Bl]

#         _, idx_large = sample_farthest_points(
#             xyz_large,
#             lengths=lengths_large,
#             K=S,
#             random_start_point=False,
#         )  # [Bl, S]

#         idx_out[large_ids] = idx_large

#     batch_ids = torch.arange(B, device=device)[:, None]
#     out = points_padded[batch_ids, idx_out]     # [B, S, C]

#     return (out, idx_out) if return_index else out

# def fps_p3d_batch_from_list(
#     points_list,
#     n_samples: int,
#     start_idx=None,
#     return_index: bool = True,
#     pad_value: float = 0.0,
# ):
#     """
#     points_list: list of [Ni, C]
#     返回:
#         out     : [B, S, C]
#         idx     : [B, S]
#         lengths : [B]
#     """
#     padded, lengths = pad_pointcloud_list(points_list, pad_value=pad_value)
#     ret = fps_p3d_batch_padded(
#         padded,
#         lengths,
#         n_samples=n_samples,
#         start_idx=start_idx,
#         return_index=return_index,
#     )
#     if return_index:
#         out, idx = ret
#         return out, idx, lengths
#     else:
#         out = ret
#         return out, lengths
    
# def voxel_downsample_keep_one_torch(points: torch.Tensor, voxel_size: float):
#     """
#     确定性体素降采样：每个 voxel 保留一个真实点（按原始顺序的第一个点）
#     points: [N, C], 前3维是 xyz
#     return:
#         ds_points: [M, C]
#         keep_idx:  [M] 在原 points 中的索引
#     """
#     assert points.ndim == 2 and points.shape[1] >= 3
#     assert voxel_size > 0

#     device = points.device
#     xyz = points[:, :3]
#     if xyz.dtype != torch.float32:
#         xyz = xyz.float()

#     # 体素坐标
#     voxel = torch.floor(xyz / voxel_size).to(torch.int64)   # [N, 3]

#     # 平移到非负，便于做唯一编码
#     voxel_min = voxel.min(dim=0).values
#     voxel = voxel - voxel_min

#     # 计算每一维跨度，做无碰撞编码
#     span = voxel.max(dim=0).values + 1  # [3]
#     sx, sy, sz = span.tolist()

#     # 对你当前这种局部 crop 点云规模，这样编码一般是安全的
#     voxel_id = voxel[:, 0] * (sy * sz) + voxel[:, 1] * sz + voxel[:, 2]   # [N]

#     # 关键点：为了确定性，按 (voxel_id, 原始索引) 排序
#     # 这样同一 voxel 内一定保留原始顺序最靠前的那个点
#     idx = torch.arange(points.shape[0], device=device, dtype=torch.int64)
#     key = voxel_id * (points.shape[0] + 1) + idx
#     order = torch.argsort(key)

#     voxel_sorted = voxel_id[order]
#     keep_mask = torch.ones_like(voxel_sorted, dtype=torch.bool)
#     keep_mask[1:] = voxel_sorted[1:] != voxel_sorted[:-1]

#     keep_idx = order[keep_mask]

#     # 恢复到原始顺序，保持输出稳定
#     keep_idx = torch.sort(keep_idx).values
#     ds_points = points[keep_idx]
#     return ds_points, keep_idx


# def pre_voxel_downsample_for_fps(
#     points: torch.Tensor,
#     cap: int = 2200,
#     trigger: int = 2400,
#     voxel_size0: float = 0.0025,
#     growth: float = 1.35,
#     max_iter: int = 6,
# ):
#     """
#     只对超大点集做保守体素预采样，保持可复现。
#     - N <= trigger: 原样返回
#     - N > trigger : 逐步增大 voxel_size，直到点数降到 cap 附近
#     """
#     assert points.ndim == 2 and points.shape[1] >= 3
#     N = points.shape[0]
#     device = points.device

#     if N <= trigger:
#         idx = torch.arange(N, device=device, dtype=torch.long)
#         return points, idx, 0.0

#     xyz = points[:, :3]
#     if xyz.dtype != torch.float32:
#         xyz = xyz.float()

#     # 用 bbox 对角线给一个保守初值，避免不同 crop 尺度差异太大
#     bbox_min = xyz.min(dim=0).values
#     bbox_max = xyz.max(dim=0).values
#     diag = torch.linalg.norm(bbox_max - bbox_min).item()

#     voxel_size = max(voxel_size0, diag / 128.0)

#     best_points = points
#     best_idx = torch.arange(N, device=device, dtype=torch.long)

#     for _ in range(max_iter):
#         ds_points, ds_idx = voxel_downsample_keep_one_torch(points, voxel_size)
#         best_points, best_idx = ds_points, ds_idx

#         if ds_points.shape[0] <= cap:
#             break

#         voxel_size *= growth

#     # 如果体素后仍然太多，再做一个确定性的均匀索引抽样兜底
#     # 这里只是极端情况兜底，正常一般到不了这一步
#     if best_points.shape[0] > cap:
#         sel = torch.linspace(
#             0, best_points.shape[0] - 1, steps=cap, device=device
#         ).long()
#         best_idx = best_idx[sel]
#         best_points = points[best_idx]

#     return best_points, best_idx, voxel_size

# def furthest_point_sampling_v2(points, colors=None, semantics=None, n_samples=4096):
#     """
#     points: [N, 3] tensor containing the whole point cloud
#     n_samples: samples you want in the sampled point cloud typically << N
#     """
#     # Convert points to PyTorch tensor if not already and move to GPU
#     # points = torch.from_numpy(points).float().to(device='cuda')  # Automatically move to GPU if needed
    
#     if colors is not None:
#         colors = torch.Tensor(colors).to(device='cuda')  # Move colors to GPU
#     if semantics is not None:
#         semantics = torch.Tensor(semantics.astype(np.int32)).to(device='cuda')  # Move semantics to GPU

#     # Number of points
#     num_points = points.shape[0]  # N

#     # Initialize an array for the sampled indices
#     sample_inds = torch.zeros(n_samples, dtype=torch.long, device='cuda')  # [S]

#     # Initialize distances to inf
#     dists = torch.ones(num_points, device='cuda') * float("inf")  # [N]

#     # Select the first point randomly
#     selected = torch.randint(num_points, (1,), dtype=torch.long, device='cuda')  # [1]
#     sample_inds[0] = selected

#     # Iteratively select points for a maximum of n_samples
#     for i in range(1, n_samples):
#         # Find the distance to the last added point in selected
#         last_added = sample_inds[i - 1]  # Scalar
#         dist_to_last_added_point = torch.sum((points[last_added] - points) ** 2, dim=-1)  # [N]

#         # If closer, update distances
#         dists = torch.min(dist_to_last_added_point, dists)  # [N]

#         # Pick the one that has the largest distance to its nearest neighbor in the sampled set
#         selected = torch.argmax(dists)  # Scalar
#         sample_inds[i] = selected

#     # Return the sampled points and corresponding attributes
#     if colors is not None and semantics is not None:
#         return (
#             points[sample_inds].cpu().numpy(),
#             colors[sample_inds].cpu().numpy(),
#             semantics[sample_inds].cpu().numpy(),
#         )  # [S, 3]
#     elif colors is not None:
#         return points[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy()
#     else:
#         return points[sample_inds].detach().cpu().numpy(), sample_inds

# def write_ply(points, filename):
#     """
#     save 3D-points and colors into ply file.
#     points: [N, 3] (X, Y, Z)
#     filename: output filename
#     """
#     # combine vertices and colors
#     vertices = np.array(
#         [tuple(point) for point in points],
#         dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
#     )

#     el = PlyElement.describe(vertices, "vertex")

#     # save PLY file
#     PlyData([el], text=True).write(filename)

# def write_npy(points, filename):
#     """
#     points: np.asarry
#     filename: output filename
#     """
#     np.save(filename, points)
#     # print(f"Saved: {filename}") 

# def is_in_workplace(env,obj_num):
#     is_in_workplace = True

#     pos, _, _ = env.obj_info(obj_num)
#     if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
#         or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
#         is_in_workplace = False
#         print(f"\033[031m Target objects {obj_num} are not in the scene!\033[0m")
  
#     return is_in_workplace

# def grasp_pcd():

#     finger1 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.015, depth=0.05)
#     finger1.translate([-0.011, -0.0575 , -0.05])

#     finger2 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.015, depth=0.05)
#     finger2.translate([-0.011, 0.0425 , -0.05])

#     finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.001)
#     finger3.translate([-0.011, -0.0575 , -0.05])
#     # finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.015)
#     # finger3.translate([-0.011, -0.0575 , -0.065])

#     gripper_mesh = finger1 + finger2 + finger3

#     gripper_pcd = gripper_mesh.sample_points_poisson_disk(200) 

#     gripper_points = torch.from_numpy(np.asarray(gripper_pcd.points)).float()

#     return gripper_points, gripper_pcd

# def hight_grasp_pcd():

#     finger1 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.024, depth=0.0475)
#     finger1.translate([-0.01, -0.0925 , -0.0475])

#     finger2 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.024, depth=0.0475)
#     finger2.translate([-0.01, 0.0685 , -0.0475])

#     finger3 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.185, depth=0.05)
#     # finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.001)
#     # finger3.translate([-0.011, -0.0575 , -0.05])
#     finger3.translate([-0.01, -0.0925 , -0.0975])

#     gripper_mesh = finger1 + finger2 + finger3
#     # pcd = o3d.geometry.PointCloud()
#     # pcd.points = o3d.utility.Vector3dVector(points[sample_inds].cpu().numpy())
#     # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#     # o3d.visualization.draw_geometries([gripper_mesh, frame])
#     gripper_pcd = gripper_mesh.sample_points_uniformly(600) 
#     # gripper_pcd = gripper_mesh.sample_points_poisson_disk(600) 

#     gripper_points = torch.from_numpy(np.asarray(gripper_pcd.points)).float()

#     # gripper_pcd.points = o3d.utility.Vector3dVector(gripper_points)

#     return gripper_points

# def push_gripper_pcd(z, n_target, oversample=2000, seed=55926):
#     """
#     gripper:[1,0,0]
#     """
#     finger1 = o3d.geometry.TriangleMesh.create_box(width=0.044, height=0.03, depth=0.05)
#     finger1.translate([0.5 - 0.022, 0 - 0.015, z])
#     # finger2 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.015, depth=0.05)
#     # finger2.translate([0.5, -0.015, z])
#     gripper_mesh = finger1 
#     pts_dense = _area_weighted_sample_on_mesh(gripper_mesh, n_samples=oversample, seed=seed)  # [M,3] CPU

#     # 2) 确定版 FPS 下采样
#     pts_final = furthest_point_sampling_det(pts_dense, n_samples=n_target, start_idx=0)  # [n_target,3]

#     # 3) Open3D点云（如需可视化）
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts_final.detach().cpu().numpy())

#     return pts_final.float(), pcd

# def TransformPCD2EndLink(point_cloud_base, pose):
#     """
#         将基坐标系下的点云变换到夹爪坐标系下。

#         Parameters:
#         - point_cloud_base: (N, 3) numpy array，基坐标系下的点云
#         - T_base_to_gripper: (4, 4) numpy array，base -> gripper 的变换矩阵

#         Returns:
#         - point_cloud_gripper: (N, 3) numpy array，夹爪坐标系下的点云
#     """
#     # 构造齐次坐标 (N, 4)
#     assert point_cloud_base.shape[1] == 3
#     assert pose.shape[0] == 7

#     device = point_cloud_base.device
#     dtype = point_cloud_base.dtype
#     # pose = torch.from_numpy(pose).float()
#     # 提取旋转和平移
#     position = pose[:3]  # [3]
#     quat = pose[3:]      # [4]

#     # 将四元数转换为旋转矩阵 (使用 torch 实现)
#     qx, qy, qz, qw = quat
#     R_mat = torch.tensor([
#         [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
#         [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
#         [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
#     ], dtype=dtype, device=device)  # [3, 3]

#     # 构造变换矩阵 T_base_to_gripper
#     T = torch.eye(4, dtype=dtype, device=device)
#     T[:3, :3] = R_mat
#     T[:3, 3] = position

#     # 计算逆变换 T_gripper_to_base
#     T_inv = torch.linalg.inv(T)  # [4, 4]

#     # 齐次点云
#     N = point_cloud_base.shape[0]
#     ones = torch.ones((N, 1), dtype=dtype, device=device)
#     points_homo = torch.cat([point_cloud_base, ones], dim=1)  # [N, 4]

#     # 变换到 gripper 坐标系
#     points_transformed = (T_inv @ points_homo.T).T  # [N, 4]
#     return points_transformed[:, :3]

# def Transform_Push2Fixed_point(global_pc, obj_pc, fixed_point, push_action):
#     """
#     All push actions must be normalized to a fixed reference point. 
#     This ensures consistent left-to-right movement by the robot, which simplifies the learning process.
#     """
#     # push_pose = np.eye(4)
#     # push_pose[:3,3] = push_action[:3]
#     # push_pose[:3,:3] = R.from_quat(push_action[3:]).as_matrix()
#     push_pose = torch.from_numpy(push_pose).float()
#     fixed_pose = torch.eye(4)
#     z = push_action[2]
#     z = z.unsqueeze(-1)
#     fixed_pose[:3,3] = torch.cat([fixed_point, z],dim=-1)
#     fixed_pose[:3,:3] = torch.tensor([[0,-1,0],
#                                       [-1,0,0],
#                                       [0,0,-1]],dtype=float)
#     T_2fixed =fixed_pose @ torch.linalg.inv(push_pose)
#     obj_pc = torch.cat([obj_pc, torch.ones(obj_pc.shape[0], 1)],dim=1)
#     global_pc = torch.cat([global_pc, torch.ones(global_pc.shape[0], 1)],dim=1)
#     global_pc = (T_2fixed @ global_pc.T).T # NX4   
#     obj_pc = (T_2fixed @ obj_pc.T).T # NX4 

#     return global_pc[:,:3], obj_pc[:, :3]

# def fuse_state(global_points,gripper_pcd):
#     """
#         1. grasp pose represented by point cloud;
#         2. contact area when gripper close;
#         3. goal object;
#     """
#     finger1 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.01, depth=0.05)
#     finger1.translate([-0.011, -0.0425 , -0.05])

#     finger2 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.01, depth=0.05)
#     finger2.translate([-0.011, 0.0425 , -0.05])

#     finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.085, depth=0.001)
#     finger3.translate([-0.011, -0.0425 , -0.05])

#     gripper_mesh = finger1 + finger2 + finger3

#     grasp_pcd = gripper_mesh.sample_points_poisson_disk(200) 
#     grasp_obb = grasp_pcd.get_oriented_bounding_box()

#     global_pcd = o3d.geometry.PointCloud()
#     global_pcd.points = o3d.utility.Vector3dVector(global_points)

#     pcd_in_gripper = global_pcd.crop(grasp_obb)

#     points_in_gripper = np.asarray(pcd_in_gripper.points)

#     fuse_points = np.vstack([points_in_gripper, gripper_pcd])

#     return fuse_points

# def fuse_state_torch(global_points: torch.Tensor,
#                      obj_points: torch.Tensor,
#                      gripper_pcd: torch.Tensor,
#                      threshold: float = 0.0065) -> torch.Tensor:
#     """
#     将 gripper 区域内与 obj 接触不充分的 global 点提取出来，与 gripper + obj 融合作为最终状态输入。
    
#     Args:
#         global_points: [N1, 3] 点云 (torch.Tensor)
#         obj_points: [N2, 3] 点云 (torch.Tensor)
#         gripper_pcd: [N3, 3] 点云 (torch.Tensor)
#         threshold: float，距离阈值（用于判定 global 点是否接触 obj）

#     Returns:
#         fuse_points: [N_fuse, 3] torch.Tensor
#     """
#     device = global_points.device
#     dtype = global_points.dtype

#     # === 定义 Gripper 包围盒（仿 Open3D 结构） ===
#     # 简化处理：统一用 AABB [x_min, x_max], [y_min, y_max], [z_min, z_max]
#     # 模拟三个 finger 包围范围
#     gripper_point = gripper_pcd.detach().cpu().numpy()
#     gripper_pc = o3d.geometry.PointCloud()
#     gripper_pc.points = o3d.utility.Vector3dVector(gripper_point)
#     gripper_obb = gripper_pc.get_oriented_bounding_box()
#     global_points = global_points.detach().cpu().numpy()
#     global_pcd = o3d.geometry.PointCloud()
#     global_pcd.points = o3d.utility.Vector3dVector(global_points)
#     points_in_gripper = global_pcd.crop(gripper_obb)
#     points_in_gripper = torch.from_numpy(np.asarray(points_in_gripper.points)).float()

#     # x_range = [-0.02, 0.02]
#     # y_range = [-0.05, 0.05]
#     # z_range = [-0.085, 0.005]  # 向上留一点 margin

#     # mask_x = (global_points[:, 0] >= x_range[0]) & (global_points[:, 0] <= x_range[1])
#     # mask_y = (global_points[:, 1] >= y_range[0]) & (global_points[:, 1] <= y_range[1])
#     # mask_z = (global_points[:, 2] >= z_range[0]) & (global_points[:, 2] <= z_range[1])

#     # mask = mask_x & mask_y & mask_z
#     # points_in_gripper = global_points[mask]  # [M, 3]

#     # === 判断 points_in_gripper 是否接触到 obj_points（欧氏距离 < threshold）===
#     if points_in_gripper.shape[0] == 0:
#         contactless_points = torch.empty((0, 3), device=device, dtype=dtype)
#     else:
#         dist = torch.cdist(points_in_gripper.unsqueeze(0), obj_points.unsqueeze(0)).squeeze(0)  # [M, N2]
#         min_dist, _ = torch.min(dist, dim=1)  # [M]
#         mask_contactless = min_dist > threshold
#         contactless_points = points_in_gripper[mask_contactless]  # [K, 3]
        

#     # === 融合为最终状态 ===
#     # fuse_points = torch.cat([contactless_points, gripper_pcd, obj_points], dim=0)  # [N_fuse, 3]
#     fuse_points = torch.cat([contactless_points, obj_points], dim=0)  # [N_fuse, 3]
#     return fuse_points

# def extend_obb_single_dir_along_global_z(pcd: o3d.geometry.PointCloud, factor: float = 10.0):
#     """
#     将 pcd 的 OBB 沿全局 Z 轴反方向单侧延长为原来的 factor 倍。
#     仅改变对应那一条边长度，保持 OBB 的朝向与另外两条边长度不变。
#     """
#     assert factor >= 1.0, "factor 应 >= 1.0（延长）"

#     # 1) 原始 OBB
#     obb = pcd.get_oriented_bounding_box()

#     R = obb.R.copy()                # 3x3，列为 OBB 局部轴（单位向量）
#     extent = obb.extent.copy()      # [ex, ey, ez]，分别对应 R 的三列
#     center = obb.center.copy()

#     # 2) 找到与全局 Z 轴最接近的 OBB 局部轴
#     z = np.array([0.0, 0.0, 1.0])   # 全局 Z 轴
#     dots = np.array([np.dot(R[:, i], z) for i in range(3)])     # 与 Z 的对齐度
#     k = int(np.argmax(np.abs(dots)))                             # 最接近 Z 的那一列索引（0/1/2）

#     old_len = extent[k]
#     new_len = factor * old_len
#     delta = new_len - old_len       # 需要增加的长度

#     # 3) 单方向：沿全局 Z 的反方向（-Z）
#     #    需要判断 R[:,k] 相对 Z 的朝向，选择与 -Z 同向的局部方向。
#     #    若 R[:,k]·Z > 0，则 -Z 与 -R[:,k] 同向；否则与 +R[:,k] 同向。
#     if dots[k] > 0:
#         dir_vec = -R[:, k]
#     else:
#         dir_vec =  R[:, k]

#     # 4) 调整中心与长度（只把“底部”那一面往 dir_vec 拉开，顶部保持不动）
#     center = center + 0.5 * delta * dir_vec
#     extent[k] = new_len

#     # 5) 生成新的 OBB
#     new_obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
#     new_obb.color = (1, 0, 0)  # 红色：延长后的 OBB
#     obb.color = (0, 0, 1)      # 蓝色：原 OBB

#     return obb, new_obb

# def fuse_state_torch_v2(global_points: torch.Tensor,
#                      gripper_pcd: torch.Tensor,
#                      threshold: float = 0.0065,
#                      ):

#     device = global_points.device
#     dtype = global_points.dtype

#     gripper_point = gripper_pcd.detach().cpu().numpy()
#     gripper_pc = o3d.geometry.PointCloud()
#     gripper_pc.points = o3d.utility.Vector3dVector(gripper_point)

#     # gripper_obb = gripper_pc.get_oriented_bounding_box()
#     _,gripper_obb = extend_obb_single_dir_along_global_z(gripper_pc)

#     global_points = global_points.detach().cpu().numpy()
#     global_pcd = o3d.geometry.PointCloud()
#     global_pcd.points = o3d.utility.Vector3dVector(global_points)
#     points_in_gripper = global_pcd.crop(gripper_obb)
#     points_in_grippers = torch.from_numpy(np.asarray(points_in_gripper.points)).to(device=device, dtype=dtype)

#     # world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
#     # o3d.visualization.draw_geometries([world, gripper_pc, global_pcd, gripper_obb])
#     # o3d.visualization.draw_geometries([world, gripper_pc, points_in_gripper])
#     # o3d.visualization.draw_geometries([world, obj_pcd,points_without_obj])
#     # o3d.visualization.draw_geometries([world, gripper_pc, obj_pcd,points_without_obj])
#     # o3d.visualization.draw_geometries([world, gripper_pc, points_in_gripper_without_obj,obj_pcd_within_gripper])

#     # points_in_gripper = torch.from_numpy(np.asarray(points_in_gripper.points)).float()

#     fuse_points = points_in_grippers

#     return fuse_points

# def fuse_state_torch_v3(global_points: torch.Tensor,
#                         gripper_pcd: torch.Tensor,
#                         threshold: float = 0.0065,
#                         ):
#     """
#     global_points: (N, 3) torch.Tensor
#     gripper_pcd:   (M, 3) torch.Tensor
#     返回:
#         fuse_points:   (K, 3) 裁剪后的点（仍在 global_points 坐标系）
#         crop_indices:  (K,)   这些点在 global_points 中的索引
#     """
#     device = global_points.device
#     dtype  = global_points.dtype

#     # ---- 构造 gripper 点云 & OBB ----
#     gripper_np = gripper_pcd.detach().cpu().numpy()
#     gripper_pc = o3d.geometry.PointCloud()
#     gripper_pc.points = o3d.utility.Vector3dVector(gripper_np)

#     # 你的扩展 OBB 函数
#     _, gripper_obb = extend_obb_single_dir_along_global_z(gripper_pc)

#     # ---- 构造 global 点云 ----
#     global_np = global_points.detach().cpu().numpy()
#     global_pcd = o3d.geometry.PointCloud()
#     global_pcd.points = o3d.utility.Vector3dVector(global_np)

#     # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#     # o3d.visualization.draw_geometries([gripper_pc, global_pcd, frame])
#     # 关键：直接拿“在 OBB 内部的点索引”
#     idx_in_obb = gripper_obb.get_point_indices_within_bounding_box(global_pcd.points)
#     # idx_in_obb: Python list[int]，对应 global_np 的行索引

#     if len(idx_in_obb) == 0:
#         # 没有点落在 OBB 内，返回空
#         crop_indices = torch.empty(0, dtype=torch.long, device=device)
#         fuse_points  = global_points[crop_indices]   # (0, 3)
#         return fuse_points, crop_indices

#     crop_indices = torch.as_tensor(idx_in_obb, dtype=torch.long, device=device)  # (K,)
#     fuse_points  = global_points[crop_indices]  # 直接从原始 Tensor 按索引取，避免再来回转换

#     return fuse_points, crop_indices

# def fuse_state(global_points: torch.Tensor,
#                      gripper_pcd: torch.Tensor,) -> torch.Tensor:
#     """
#     将 gripper 区域内与 obj 接触不充分的 global 点提取出来，与 gripper + obj 融合作为最终状态输入。
    
#     Args:
#         global_points: [N1, 3] 点云 (torch.Tensor)
#         gripper_pcd: [N3, 3] 点云 (torch.Tensor)

#     Returns:
#         fuse_points: [N_fuse, 3] torch.Tensor
#     """
#     device = global_points.device
#     dtype = global_points.dtype

#     # x_range = [-0.022, 0.022]
#     # y_range = [-0.05, 0.05]
#     # z_range = [-0.085, 0.005] 
    
#     x_range = [-0.011, 0.011]
#     y_range = [-0.0425, 0.0525]
#     z_range = [-0.05, 0.0] # 向上留一点 margin

#     mask_x = (global_points[:, 0] >= x_range[0]) & (global_points[:, 0] <= x_range[1])
#     mask_y = (global_points[:, 1] >= y_range[0]) & (global_points[:, 1] <= y_range[1])
#     mask_z = (global_points[:, 2] >= z_range[0]) & (global_points[:, 2] <= z_range[1])

#     mask = mask_x & mask_y & mask_z
#     points_in_gripper = global_points[mask]  # [M, 3]

#     fuse_points = torch.cat([points_in_gripper, gripper_pcd], dim=0)  # [N_fuse, 3]
#     return fuse_points

# def natural_key(s):
#     # 提取字符串中的整数用于排序，例如 "ply_global_10.ply" -> ['ply_global_', 10, '.ply']
#     return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

# def uniform_point_count(points, target_n=345, jitter_std=0.00001):
#     """
#     points: np.ndarray of shape (N, 3)
#     Returns: np.ndarray of shape (target_n, 3)
#     """
#     N = points.shape[0]
    
#     if N > target_n:
#         # 下采样：随机采样
#         idx = np.random.choice(N, target_n, replace=False)
#         return points[idx]
    
#     elif N < target_n:
#         # 上采样：重复采样 + 可选扰动
#         idx = np.random.choice(N, target_n, replace=True)
#         sampled = points[idx]
        
#         # 可选：添加微小扰动，防止重复点完全重合
#         noise = np.random.normal(0, jitter_std, size=sampled.shape)
#         return sampled + noise
    
#     else:
#         return points
    
# import torch

# def uniform_point_count_torch(points: torch.Tensor, target_n: int = 240, jitter_std: float = 1e-6) -> torch.Tensor:
#     """
#     对点云进行上/下采样，使其统一为 target_n 个点，适用于 torch.Tensor 输入。

#     Args:
#         points: [N, 3] torch.Tensor，原始点云
#         target_n: int，目标点数
#         jitter_std: float，加性噪声标准差（仅用于上采样时防止重合）

#     Returns:
#         [target_n, 3] torch.Tensor，重采样后的点云
#     """
#     N = points.shape[0]
#     device = points.device
#     dtype = points.dtype

#     if N > target_n:
#         # 下采样
#         idx = torch.randperm(N, device=device)[:target_n]
#         return points[idx]

#     elif N < target_n:
#         # 上采样
#         idx = torch.randint(0, N, (target_n,), device=device)
#         sampled = points[idx]
#         if jitter_std > 0:
#             noise = torch.randn_like(sampled) * jitter_std
#             sampled = sampled + noise
#         return sampled

#     else:
#         return points

# def pc_normalize(pc: torch.Tensor):
#     """
#     输入:
#         pc: Tensor of shape [N, 3]，点云坐标，类型为 float32 或 float64
#     返回:
#         归一化后的点云，中心位于原点，最大半径为1
#     """
#     centroid = torch.mean(pc, dim=1)          # [3]
#     pc = pc - centroid                         # 平移到原点
#     m = torch.max(torch.sqrt(torch.sum(pc**2, dim=2)))  # 最远点距离
#     pc = pc / m                                # 缩放归一化
#     return pc

# def pc_normalize_grasp(pc: torch.Tensor):
#     """
#     输入:
#         pc: Tensor of shape [N, 3]，点云坐标，类型为 float32 或 float64
#     返回:
#         归一化后的点云，中心位于原点，最大半径为1
#     """
#     centroid = torch.mean(pc, dim=0)          # [3]
#     pc = pc - centroid                         # 平移到原点
#     m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))  # 最远点距离
#     pc = pc / m                                # 缩放归一化
#     return pc, centroid, m

# def pc_normalize_for_obj(pc: torch.Tensor,
#                         centroid: torch.Tensor,
#                         m: torch.Tensor):
#     """
#     obj pc should used the same normalization way in global pc.
#     """
#     pc = pc - centroid
#     pc = pc / m
#     return pc

# #-----------sample push action----------------------
# def transform_points_to_camera(points_world, T_cam_base):
#     num_points = points_world.shape[0]
#     homo_points = np.hstack((points_world, np.ones((num_points, 1))))  # Nx4
#     points_cam = (T_cam_base @ homo_points.T).T[:, :3]  # Nx3
#     return points_cam

# def project_points_to_image(points_cam, fx, fy, cx, cy):
#     X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
#     Z[Z <= 0] = 1e-6  
#     u = (X * fx / Z + cx).astype(int)
#     v = (Y * fy / Z + cy).astype(int)
#     return u, v

# def dilate_masks(masks, kernel_size=3, iterations=1):
#     H, W = masks.shape
#     object_ids = np.unique(masks)
#     object_ids = object_ids[object_ids != 0] # 排除背景

#     dilated_mask = np.zeros_like(masks)

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

#     for obj_id in object_ids:
#         binary_mask = (masks == obj_id).astype(np.uint8)
#         dilated = cv2.dilate(binary_mask, kernel, iterations=iterations)
#         dilated_mask[dilated > 0] = obj_id

#     return dilated_mask

# def segment_pointcloud_by_mask(points, masks):
#     masks = dilate_masks(masks)
#     intrinsics = np.array([[630000.0, 0, 320], [0, 630000.0, 240], [0, 0, 1]])  
#     fx, fy = intrinsics[0, 0], intrinsics[1, 1]
#     cx, cy = intrinsics[0, 2], intrinsics[1, 2]

#     position = np.array([0.5, 0, 1000.0]) 
#     rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))
#     rot_matrix = np.array(p.getMatrixFromQuaternion(rotation)).reshape(3, 3)

#     T_base_to_cam = np.eye(4)
#     T_base_to_cam[:3, :3] = rot_matrix
#     T_base_to_cam[:3, 3] = position
#     T_cam_base = np.linalg.inv(T_base_to_cam)

#     # points = np.asarray(scene_pcd.points)
#     H, W = masks.shape

#     points_cam = transform_points_to_camera(points, T_cam_base)
#     u, v = project_points_to_image(points_cam, fx, fy, cx, cy)

#     valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
#     u_valid = u[valid]
#     v_valid = v[valid]
#     points_valid = points[valid]
#     labels = masks[v_valid, u_valid]

#     object_ids = np.unique(labels)
#     object_ids = object_ids[object_ids > 0]

#     object_pcds = []
#     object_masks = []
#     for obj_id in object_ids[1:]:
#         idx = np.where(labels == obj_id)[0]
#         obj_points = points_valid[idx]

#         obj_pcd = o3d.geometry.PointCloud()
#         obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
#         object_pcds.append(obj_pcd)

#         mask = np.zeros((H,W),dtype=bool)
#         mask[v_valid[idx],u_valid[idx]] = True
#         object_masks.append(mask)

#     return object_pcds,object_masks

# def sample_surface_points(object_pcd,expand=0.016,step=0.03):
#     points = np.asarray(object_pcd.points)
#     aabb = object_pcd.get_axis_aligned_bounding_box()
#     z_mean = (aabb.get_max_bound()[2] + aabb.get_min_bound()[2]) / 2
#     xy = points[:, :2]
#     resolution = 0.001
#     xy_min = xy.min(axis=0)
#     xy_min = xy.min(axis=0)
#     xy_max = xy.max(axis=0)
#     pad = 10
#     img_size = np.ceil((xy_max - xy_min) / resolution).astype(int) + 2*pad
#     img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
#     indices = ((xy - xy_min) / resolution).astype(int) + pad
#     img[indices[:, 1], indices[:, 0]] = 255 
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
#     closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
#     img_filled = binary_fill_holes(closed>0).astype(np.uint8) * 255
#     dilated_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     dilated = cv2.dilate(img_filled, dilated_kernel, iterations=1)
#     contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) == 0:
#         raise ValueError("No contour found from projected point cloud.")
#     img_color = np.zeros((img_filled.shape[0],img_filled.shape[1],3),dtype=np.uint8)
#     img_color[img_filled > 0] = [0,0,255]
#     max_contour = max(contours, key=cv2.contourArea)
#     # cv2.drawContours(img_color,max_contour,-1,(255,0,0),1)
#     # cv2.imshow('Contours',img_color)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     contour_pts_image = max_contour[:, 0, :].astype(np.float32) 
#     contour_pts_image -= pad
#     contour_pts_world = contour_pts_image * resolution + xy_min
#     poly = Polygon(contour_pts_world)
#     offset_polygon = poly.buffer(expand)
#     if isinstance(offset_polygon,MultiPolygon):
#         offset_polygon = max(offset_polygon.geoms, key = lambda p: p.area)
#     boundary_coords = np.array(offset_polygon.exterior.coords[:-1])  
#     push_xy = interpolate_polygon_edges_with_step(boundary_coords,step)

#     [0.276, 0.724], [-0.224, 0.224]
#     sampled_points = []
#     for uv_pt in push_xy:
#         if 0.276 < uv_pt[0] < 0.724 and -0.224 < uv_pt[1] < 0.224:
#             sampled_points.append([uv_pt[0], uv_pt[1], z_mean])

#     return np.array(sampled_points)


# def interpolate_polygon_edges_with_step(hull_pts, step=0.005):
#     hull_pts = np.asarray(hull_pts, dtype=np.float32)
#     n = len(hull_pts)
#     if n < 2:
#         return hull_pts.copy()

#     # Step 1: 计算每条边的向量和长度
#     edges = hull_pts[(np.arange(n) + 1) % n] - hull_pts # shape [n, 2]
#     edge_lengths = np.linalg.norm(edges, axis=1)
#     total_length = np.sum(edge_lengths)

#     if total_length < 1e-6:
#         return hull_pts[:1] # 全部重合时只返回一个点

#     # Step 2: 计算累积长度
#     cumulative_lengths = np.cumsum(edge_lengths)
#     num_samples = max(int(np.floor(total_length / step)), 1)
#     sample_distances = np.linspace(0, total_length, num_samples, endpoint=False)

#     # Step 3: 插值采样点
#     sampled_pts = []
#     edge_idx = 0
#     curr_edge_start = hull_pts[0]
#     curr_edge_vec = edges[0]
#     curr_edge_len = edge_lengths[0]
#     curr_cum_len = 0.0

#     for d in sample_distances:
#     # 移动到对应边
#         while d >= cumulative_lengths[edge_idx]:
#             curr_cum_len = cumulative_lengths[edge_idx]
#             edge_idx = (edge_idx + 1) % n
#             curr_edge_start = hull_pts[edge_idx]
#             curr_edge_vec = edges[edge_idx]
#             curr_edge_len = edge_lengths[edge_idx]

#         t = (d - curr_cum_len) / curr_edge_len # 当前边上的归一化位置
#         pt = curr_edge_start + curr_edge_vec * t
#         sampled_pts.append(pt)

#     return np.array(sampled_pts)



# def remove_points_near_other_cloud(pcd_A, pcd_B, radius):
#     # A_points = np.asarray(pcd_A.points)
#     # B_points = np.asarray(pcd_B.points)
#     # B_tree = cKDTree(B_points[:, :2])
#     # # 查询A点云中每个点在半径内的邻近点
#     # neighbors = B_tree.query_ball_point(A_points[:, :2], r=radius, return_length=True)
#     # # 没有邻近点的才保留
#     # keep_mask = np.array(neighbors) == 0
#     # filtered_pcd_A = o3d.geometry.PointCloud()
#     # filtered_pcd_A.points = o3d.utility.Vector3dVector(A_points[keep_mask])

#     # return filtered_pcd_A
#     A_points = np.asarray(pcd_A.points)
#     B_points = np.asarray(pcd_B.points)

#     B_tree = cKDTree(B_points[:, :2])

#     keep_mask = []
#     for i in range(len(A_points)):
#         a_xy = A_points[i, :2]
#         a_z = A_points[i, 2]
#         idxs = B_tree.query_ball_point(a_xy, r=radius)

#         keep = True
#         for j in idxs:
#             if B_points[j, 2] > a_z:
#                 keep = False
#                 break
#         keep_mask.append(keep)

#     keep_mask = np.array(keep_mask)
#     filtered_pcd_A = o3d.geometry.PointCloud()
#     filtered_pcd_A.points = o3d.utility.Vector3dVector(A_points[keep_mask])

#     return filtered_pcd_A

# def compute_pose_dict(pcd_a,pcd_b):
#     """
#     计算点云A中每个点的姿态，返回每个点的pose字典
    
#     Args:
#     pcd_a: 点云A (open3d.geometry.PointCloud)
#     pcd_b: 点云B (open3d.geometry.PointCloud)

#     Returns:
#     poses_dict: 包含每个点姿态的字典，键为点云A中的点索引，值为对应的4x4姿态矩阵
#     """
#     # 计算点云B的质心
#     def compute_centroid(pcd):
#         points = np.asarray(pcd.points)
#         centroid = np.mean(points, axis=0)
#         return centroid

#     # 计算点云A中每个点的姿态
#     def compute_pose(point, centroid_b):
#         # 计算点到质心的向量
#         direction = centroid_b - point
#         # 投影到xy平面
#         direction_xy = direction[:2]
#         direction_xy /= np.linalg.norm(direction_xy)  # 归一化
        
#         # x轴方向
#         x_axis = np.array([direction_xy[0], direction_xy[1], 0])
#         # y轴方向可以取竖直方向
#         z_axis = np.array([0, 0, -1])  # 固定为竖直向下
#         # z轴方向为x轴和y轴的叉积
#         y_axis = np.cross(z_axis, x_axis)
#         y_axis /= np.linalg.norm(y_axis)

#         # 计算旋转矩阵
#         rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

#         # 组合位姿
#         pose = np.eye(4)
#         pose[:3, :3] = rotation_matrix
#         pose[:3, 3] = point

#         return pose

#     # 获取点云B的质心
#     centroid_b = compute_centroid(pcd_b)

#     # 保存点云A每个点的pose
#     poses_dict = []

#     # 遍历点云A中的每个点，计算并保存对应的姿态
#     for idx, point in enumerate(np.asarray(pcd_a.points)):
#         pose = compute_pose(point, centroid_b)
#         poses_dict.append(pose)

#     return poses_dict

# def get_push_pose(object_pcd, pcd1):

#     sample_points = sample_surface_points(object_pcd)
#     if len(sample_points) == 0:
#        poses_dict = []
#     else:
#         sampled_pcd = o3d.geometry.PointCloud()
#         sampled_pcd.points = o3d.utility.Vector3dVector(sample_points)
#         filter_pcd = remove_points_near_other_cloud(sampled_pcd, pcd1, radius=0.015)
#         # filter_pcd = remove_points_near_other_cloud(sampled_pcd, pcd1, radius=0.001)
#         # poses_dict = compute_pose_dict(filter_pcd, object_pcd)
#         poses_dict = compute_pose_dict(filter_pcd, object_pcd)
#     return poses_dict


# def sample_push_action(points,object_pcds):
    
#     # points = np.asarray(scene_pcd.points)

#     # 找到z坐标的最小值
#     minZ = np.min(points[:, 2])

#     # 筛选z坐标大于minZ+0.005的点
#     indices = np.where(points[:, 2] >= minZ + 0.005)[0]
#     new_points = points[indices]

#     # 创建新的点云对象
#     pcd1 = o3d.geometry.PointCloud()
#     pcd1.points = o3d.utility.Vector3dVector(new_points)
#     # object_pcds,_ = segment_pointcloud_by_mask(points,masks)
#     poses_dicts = []

#     # 可视化聚类结果
#     for i, object_pcd in enumerate(object_pcds):
#         # o3d.visualization.draw_geometries([object_pcd])
#         poses_dict = get_push_pose(object_pcd, pcd1)
#         if len(poses_dict) == 0:
#             continue
#         poses_dicts.append(poses_dict)

#     # secen_pcd = o3d.geometry.PointCloud()
#     # secen_pcd.points = o3d.utility.Vector3dVector(points)
#     # vis = o3d.visualization.Visualizer()
#     # vis.create_window()
#     # vis.add_geometry(pcd1)
#     # vis.add_geometry(secen_pcd)
#     # for poses_dict in poses_dicts:  # 外层：每个聚类块
#     #     for pose in poses_dict:     # 内层：该块内每个点的 pose
#     #         coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.025)
#     #         coordinate_frame.transform(pose)
#     #         vis.add_geometry(coordinate_frame)

#     # vis.run()
#     # vis.destroy_window()
    
#     return np.vstack(poses_dicts)
# def save_grasp_action(global_pc, obj_pc, grasp_action, data_collect_id):

#     ply_global_name = f"env_data_collection/adjust_data/global_pc/global_pc_{data_collect_id:05d}.ply"
#     ply_obj_name = f"env_data_collection/adjust_data/obj_pc/ply_obj_{data_collect_id:05d}.ply"
#     write_ply(global_pc, ply_global_name)
#     write_ply(obj_pc, ply_obj_name)
#     pose = np.hstack([grasp_action[:3, 3],R.from_matrix(grasp_action[:3, :3]).as_quat()])
#     with open("env_data_collection/adjust_data/poses.txt", "a") as file:
#                 file.write(
#                     f"{pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]}"
#                     + "\n"
#                 )
    
#     with open("env_data_collection/adjust_data/labels.txt", "a") as f:
#         f.write(f"{int(0)}\n")
    
# def transform_matrix2quat(push_actions):

#     push_actions_sac = []
#     for i in range(len(push_actions)):
#         action = push_actions[i]
#         position = action[:3, 3]
#         rotation = action[:3, :3]
#         r = R.from_matrix(rotation)
#         quat = r.as_quat()
#         t = np.hstack((position, quat))
#         push_actions_sac.append(t)
#     return np.vstack(push_actions_sac)

# def transform_np2tensor(x):
#     x = torch.from_numpy(x).float()
#     x = x.unsqueeze(0) 
#     x = x.transpose(1,2)
#     return x



# def _area_weighted_sample_on_mesh(mesh: o3d.geometry.TriangleMesh, n_samples: int, seed: int = 42) -> torch.Tensor:
#     """
#     在三角网格表面按面积“确定性随机”采样 n_samples 个点（受 seed 控制）。
#     返回: [n_samples, 3] torch.float32 (CPU)
#     """
#     # 顶点与面
#     V = np.asarray(mesh.vertices)        # [Nv, 3]
#     F = np.asarray(mesh.triangles)       # [Nf, 3]

#     # 面积
#     v0 = V[F[:, 0]]
#     v1 = V[F[:, 1]]
#     v2 = V[F[:, 2]]
#     tri_areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
#     area_sum = tri_areas.sum()
#     if area_sum <= 0:
#         raise ValueError("Mesh has zero total area.")

#     # 面选择的概率分布
#     probs = tri_areas / area_sum

#     # 受控随机源
#     rng = np.random.RandomState(seed)

#     # 采样面索引
#     face_idx = rng.choice(len(F), size=n_samples, replace=True, p=probs)  # [n_samples]
#     f0 = V[F[face_idx, 0]]
#     f1 = V[F[face_idx, 1]]
#     f2 = V[F[face_idx, 2]]

#     # 三角形内的均匀采样（barycentric，sqrt trick）
#     u = rng.rand(n_samples, 1)
#     v = rng.rand(n_samples, 1)
#     su = np.sqrt(u)
#     w0 = 1.0 - su
#     w1 = su * (1.0 - v)
#     w2 = su * v

#     pts = (w0 * f0) + (w1 * f1) + (w2 * f2)  # [n_samples, 3]
#     return torch.from_numpy(pts.astype(np.float32))  # CPU tensor


# def furthest_point_sampling_det(points: torch.Tensor, n_samples: int, start_idx: int = 0) -> torch.Tensor:
#     """
#     确定版 FPS（Euclidean）。输入 points:[N,3](CPU/GPU均可)，返回 [n_samples,3]，与设备一致。
#     """
#     if not torch.is_tensor(points):
#         points = torch.tensor(points, dtype=torch.float32)
#     device = points.device
#     points = points.to(device=device, dtype=torch.float32)

#     N = points.shape[0]
#     n_samples = min(n_samples, N)

#     sample_inds = torch.empty(n_samples, dtype=torch.long, device=device)
#     dists = torch.full((N,), float("inf"), device=device)

#     selected = torch.tensor([start_idx], dtype=torch.long, device=device)
#     sample_inds[0] = selected

#     for i in range(1, n_samples):
#         last = sample_inds[i - 1]
#         dist_to_last = torch.sum((points - points[last]) ** 2, dim=-1)
#         dists = torch.minimum(dists, dist_to_last)
#         selected = torch.argmax(dists)  # 确定的tie-break：返回第一个最大值
#         sample_inds[i] = selected

#     return points[sample_inds]


# def grasp_pcd_bluenoise_like(n_target: int = 500, oversample: int = 5000, seed: int = 42, extend=0):
#     """
#     生成“更均匀、可复现”的抓手点云：
#     1) 构网格 → 2) 面均匀过采样 oversample → 3) FPS 取 n_target → 4) 返回 torch.Tensor 与 Open3D 点云
#     """
#     # --- 构造你的夹爪网格（与你现有版本一致） ---
#     finger1 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.015, depth=0.0475 + extend)
#     finger1.translate([-0.01, -0.0835, -0.0475])
#     finger2 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.015, depth=0.0475 + extend)
#     finger2.translate([-0.01, 0.0685, -0.0475])
#     finger3 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.167, depth=0.015)
#     finger3.translate([-0.01, -0.0835, -0.0625])
#     # finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.001)
#     # finger3.translate([-0.011, -0.0575 , -0.05])
#     gripper_mesh = finger1 + finger2 + finger3

#     # 1) 受控过采样
#     pts_dense = _area_weighted_sample_on_mesh(gripper_mesh, n_samples=oversample, seed=seed)  # [M,3] CPU

#     # 2) 确定版 FPS 下采样
#     pts_final = furthest_point_sampling_det(pts_dense, n_samples=n_target, start_idx=0)  # [n_target,3]

#     # 3) Open3D点云（如需可视化）
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts_final.detach().cpu().numpy())

#     return pts_final.float(), pcd

# def gripper_point_width(n_target: int = 500, oversample: int = 5000, seed: int = 42, gripper_width=None):
#     # --- 构造你的夹爪网格（与你现有版本一致） ---
#     finger1 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.0143, depth=0.0475)
#     finger1.translate([-0.01, -gripper_width/2 - 0.0143, -0.0475])
#     finger2 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.0143, depth=0.0475)
#     finger2.translate([-0.01, gripper_width/2, -0.0475])
#     finger3 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=gripper_width+0.0286, depth=0.0143)
#     finger3.translate([-0.01, -gripper_width/2 - 0.0143, -0.0618])
#     # finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.001)
#     # finger3.translate([-0.011, -0.0575 , -0.05])
#     gripper_mesh = finger1 + finger2 + finger3

#     # 1) 受控过采样
#     pts_dense = _area_weighted_sample_on_mesh(gripper_mesh, n_samples=oversample, seed=seed)  # [M,3] CPU

#     # 2) 确定版 FPS 下采样
#     pts_final = furthest_point_sampling_det(pts_dense, n_samples=n_target, start_idx=0)  # [n_target,3]

#     # 3) Open3D点云（如需可视化）
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts_final.detach().cpu().numpy())

#     return pts_final.float(), pcd

# def extend_gripper_point(n_target: int = 500, oversample: int = 5000, seed: int = 42, gripper_width=None):
#     extend = (gripper_width / 2) * math.sin(math.radians(24))
#     finger1 = o3d.geometry.TriangleMesh.create_box(width=0.02 + extend * 2, height=0.015, depth=0.0475 + 0.01)
#     finger1.translate([-0.01 - extend, -gripper_width/2 - 0.015, -0.0475  ])
#     finger2 = o3d.geometry.TriangleMesh.create_box(width=0.02 + extend * 2, height=0.015, depth=0.0475 + 0.01)
#     finger2.translate([-0.01 - extend, gripper_width/2, -0.0475 ])
#     finger3 = o3d.geometry.TriangleMesh.create_box(width=0.02 + extend * 2, height=gripper_width + 0.03, depth=0.015)
#     finger3.translate([-0.01 - extend, -gripper_width/2 - 0.015, -0.0625])
#     # finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.001)
#     # finger3.translate([-0.011, -0.0575 , -0.05])
#     gripper_mesh = finger1 + finger2 + finger3

#     # 1) 受控过采样
#     pts_dense = _area_weighted_sample_on_mesh(gripper_mesh, n_samples=oversample, seed=seed)  # [M,3] CPU

#     # 2) 确定版 FPS 下采样
#     pts_final = furthest_point_sampling_det(pts_dense, n_samples=n_target, start_idx=0)  # [n_target,3]

#     # 3) Open3D点云（如需可视化）
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts_final.detach().cpu().numpy())
#     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#     o3d.visualization.draw_geometries([pcd, frame])
#     return pts_final.float(), pcd

# def append_square_plane_voxel_cull(
#     pcd: o3d.geometry.PointCloud,
#     half_size: float = 0.2,           # 方形半边长（m），边长=2*half_size
#     spacing: float = 0.001,           # 平面网格点间距（m）
#     center_method: Literal["mean", "median"] = "mean",
#     mode: Literal["xy", "3d"] = "xy", # 体素判定模式：xy 或 3d
#     voxel_size: Union[float, Tuple[float, float, float]] = 0.002,  # 体素尺寸（m）
#     origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),          # 体素网格原点（m）
#     keep_attrs: bool = False,
#     plane_color: Optional[Tuple[float, float, float]] = (0.6, 0.6, 0.6),
#     set_plane_normals: bool = True,
# ) -> o3d.geometry.PointCloud:
#     """
#     生成位于 z=0 的正方形平面点云（规则网格），并用体素法剔除与输入点云“共用同一体素”的平面点，然后与输入点云合并返回。

#     共用体素的定义:
#       - mode="xy": floor((x - ox)/vx), floor((y - oy)/vy) 相同即视为共用体素
#       - mode="3d": 上述再加 floor((z - oz)/vz) 也相同

#     参数:
#         pcd: 输入 Open3D 点云
#         half_size: 方形半边长（m）
#         spacing: 平面点网格间距（m）
#         center_method: 确定平面中心 (cx,cy) 的方法：'mean' 或 'median'
#         mode: 'xy' 或 '3d' 体素重叠判定模式
#         voxel_size: 体素大小，可为标量或 (vx,vy,vz)
#         origin: 体素网格原点 (ox,oy,oz)。体素化与原点有关，需固定以保证一致性
#         keep_attrs: 是否为平面点补齐颜色/法向（仅当输入已具备对应属性时）
#         plane_color: 平面点颜色 [0,1]
#         set_plane_normals: 是否将平面点法向设置为 (0,0,1)

#     返回:
#         合并后的点云（输入点云 + 去重后的平面点）
#     """
#     if pcd.is_empty():
#         raise ValueError("输入点云为空")

#     pts = np.asarray(pcd.points)
#     if pts.ndim != 2 or pts.shape[1] != 3:
#         raise ValueError("点坐标应为 (N,3)")

#     # 1) 平面中心
#     if center_method == "mean":
#         cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
#     elif center_method == "median":
#         cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
#     else:
#         raise ValueError("center_method 仅支持 'mean' 或 'median'")
#     cz = 0
#     # cz = float(pts[:, 2].min())

#     # 2) 生成正方形网格平面（z=0）
#     xs = np.arange(-half_size, half_size + spacing, spacing) + cx
#     ys = np.arange(-half_size, half_size + spacing, spacing) + cy
#     xx, yy = np.meshgrid(xs, ys, indexing="xy")
#     plane_pts = np.stack([xx, yy, np.full_like(xx, cz)], axis=-1).reshape(-1, 3)  # (M,3)

#     # 3) 体素参数规范化
#     if isinstance(voxel_size, (int, float)):
#         vx = vy = vz = float(voxel_size)
#     else:
#         if len(voxel_size) != 3:
#             raise ValueError("voxel_size 必须是标量或长度为3的元组")
#         vx, vy, vz = map(float, voxel_size)

#     ox, oy, oz = origin

#     # 4) 计算输入点云的体素索引集合
#     if mode == "xy":
#         occ_ix = np.floor((pts[:, 0] - ox) / vx).astype(np.int64)
#         occ_iy = np.floor((pts[:, 1] - oy) / vy).astype(np.int64)
#         occ_indices = np.stack([occ_ix, occ_iy], axis=1)  # (N,2)
#         plane_ix = np.floor((plane_pts[:, 0] - ox) / vx).astype(np.int64)
#         plane_iy = np.floor((plane_pts[:, 1] - oy) / vy).astype(np.int64)
#         plane_indices = np.stack([plane_ix, plane_iy], axis=1)  # (M,2)
#         dim = 2
#     elif mode == "3d":
#         occ_ix = np.floor((pts[:, 0] - ox) / vx).astype(np.int64)
#         occ_iy = np.floor((pts[:, 1] - oy) / vy).astype(np.int64)
#         occ_iz = np.floor((pts[:, 2] - oz) / vz).astype(np.int64)
#         occ_indices = np.stack([occ_ix, occ_iy, occ_iz], axis=1)  # (N,3)
#         plane_ix = np.floor((plane_pts[:, 0] - ox) / vx).astype(np.int64)
#         plane_iy = np.floor((plane_pts[:, 1] - oy) / vy).astype(np.int64)
#         plane_iz = np.floor((plane_pts[:, 2] - oz) / vz).astype(np.int64)
#         plane_indices = np.stack([plane_ix, plane_iy, plane_iz], axis=1)  # (M,3)
#         dim = 3
#     else:
#         raise ValueError("mode 必须是 'xy' 或 '3d'")

#     # 5) 利用“行视图”做高速集合判定：将 (k,dim) int64 转成无类型字节视图，再 in1d
#     def _row_view(a: np.ndarray) -> np.ndarray:
#         """将 int64 矩阵的每行视为一个字节记录，用于 in1d/集合判定"""
#         if not a.flags['C_CONTIGUOUS']:
#             a = np.ascontiguousarray(a)
#         return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()

#     occ_view = _row_view(occ_indices)
#     occ_unique = np.unique(occ_view)  # 去重，减小集合规模
#     plane_view = _row_view(plane_indices)
#     overlap_mask = np.in1d(plane_view, occ_unique, assume_unique=True)  # True 表示共用体素，需要剔除
#     keep_mask = ~overlap_mask

#     plane_pts_kept = plane_pts[keep_mask]

#     # 6) 合并并返回
#     out_pts = np.concatenate([pts, plane_pts_kept], axis=0)
#     out = o3d.geometry.PointCloud()
#     out.points = o3d.utility.Vector3dVector(out_pts.astype(np.float64))

#     if keep_attrs:
#         # 颜色
#         if pcd.has_colors():
#             in_cols = np.asarray(pcd.colors)
#             if plane_color is None:
#                 plane_color = (0.5, 0.5, 0.5)
#             plane_cols = np.tile(np.array(plane_color, dtype=np.float64)[None, :],
#                                  (plane_pts_kept.shape[0], 1))
#             out_cols = np.vstack([in_cols, plane_cols])
#             out.colors = o3d.utility.Vector3dVector(out_cols)
#         # 法向
#         if pcd.has_normals():
#             in_normals = np.asarray(pcd.normals)
#             if set_plane_normals:
#                 plane_normals = np.tile(np.array([0.0, 0.0, 1.0]), (plane_pts_kept.shape[0], 1))
#             else:
#                 plane_normals = np.zeros((plane_pts_kept.shape[0], 3))
#             out_normals = np.vstack([in_normals, plane_normals])
#             out.normals = o3d.utility.Vector3dVector(out_normals)

#     return out


# def furthest_point_sampling_onehot_nocuda(points, colors=None, semantics=None, n_samples=4096, start_idx=0):
#     """
#     points: [N, 6] = [x,y,z, onehot_3]，仅 xyz 参与FPS，返回 [S, 6]
#     n_samples: samples you want in the sampled point cloud typically << N
#     """
#     # if colors is not None:
#     #     colors = torch.Tensor(colors).cuda()
#     if colors is not None:
#         colors = torch.as_tensor(colors, dtype=torch.float32, device=points.device)
#     if semantics is not None:
#         semantics = torch.as_tensor(semantics.astype(np.int32), device=points.device)

#     num_points = points.shape[0]  # N
#     # n_samples = min(n_samples, num_points)

#     sample_inds = torch.zeros(n_samples, dtype=torch.long, device=points.device)  # 保持在同device
#     dists = torch.ones(num_points, device=points.device) * float("inf")

#     # 仅用于距离计算的 xyz 视图 ------------- NEW
#     xyz = points[:, :3].contiguous()  # [N,3]

#     # 选择起点
#     # selected = torch.randint(num_points, (1,), dtype=torch.long)  # [1]
#     selected = torch.tensor(start_idx, dtype=torch.long, device=points.device)  # ---- 小修：标量
#     sample_inds[0] = selected

#     for i in range(1, n_samples):
#         last_added = sample_inds[i - 1]
#         # 用 xyz 计算距离 ---------------------- NEW
#         dist_to_last_added_point = torch.sum((xyz[last_added] - xyz) ** 2, dim=-1)
#         dists = torch.min(dist_to_last_added_point, dists)
#         selected = torch.argmax(dists)
#         sample_inds[i] = selected

#     # 返回时用原始 points 按索引切片，带回 one-hot 与任何额外通道
#     if colors is not None and semantics is not None:
#         return (
#             points[sample_inds].cpu().numpy(),     # [S,6] -------------- NEW: 原张量切片
#             colors[sample_inds].cpu().numpy(),
#             semantics[sample_inds].cpu().numpy(),
#         )
#     elif colors is not None:
#         return points[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy()
#     else:
#         return points[sample_inds]                 # [S,6]

# def furthest_point_sampling_onehot_p3d(points, colors=None, semantics=None, n_samples=4096, start_idx=0):

#     assert isinstance(points, torch.Tensor), "points 需为 torch.Tensor"
#     device = points.device
#     N = points.shape[0]
#     K = min(int(n_samples), int(N))

#     # xyz 视图 (B, P, 3)
#     xyz = points[:, :3].to(dtype=torch.float32, device=device).contiguous().unsqueeze(0)  # [1, N, 3]

#     # 若需要固定起点：把 start_idx 放到第一个位置，再跑 FPS
#     if start_idx is not None:
#         if not (0 <= start_idx < N):
#             raise ValueError(f"start_idx 超界: {start_idx} (总点数 {N})")
#         perm = torch.arange(N, device=device)
#         # 交换 perm[0] 与 perm[start_idx]
#         if start_idx != 0:
#             perm0 = perm[0].clone()
#             perm[0] = perm[start_idx]
#             perm[start_idx] = perm0
#         xyz_perm = xyz[:, perm, :]  # [1, N, 3]
#         # 运行 FPS（不随机起点）
#         _, idx_perm = sample_farthest_points(xyz_perm, K=K, random_start_point=False)
#         # idx_perm 是针对于 perm 后坐标的索引，映射回原索引：
#         sel = perm[idx_perm[0]]  # [K]
#     else:
#         # 无固定起点，直接运行
#         _, idx = sample_farthest_points(xyz, K=K, random_start_point=False)
#         sel = idx[0]  # [K]

#     # 统一把可选附加信息搬到同 device/dtype 后再切片
#     out_points = points[sel]  # [K, 6]，包含 one-hot 等附加通道

#     out_colors = None
#     out_semantics = None
#     if colors is not None:
#         out_colors = torch.as_tensor(colors, dtype=torch.float32, device=device)[sel]
#     if semantics is not None:
#         # 保持 int 类型
#         sem = torch.as_tensor(semantics, device=device)
#         if sem.dtype != torch.long:
#             sem = sem.to(torch.long)
#         out_semantics = sem[sel]

#     # 与你原函数的返回风格保持一致
#     if out_colors is not None and out_semantics is not None:
#         return out_points.cpu().numpy(), out_colors.cpu().numpy(), out_semantics.cpu().numpy()
#     elif out_colors is not None:
#         return out_points.cpu().numpy(), out_colors.cpu().numpy()
#     else:
#         return out_points

# def Transform_Push2Fixed_point_onehot(global_points_onehot: torch.Tensor,
#                                   fixed_point: torch.Tensor,
#                                   push_action: torch.Tensor) -> torch.Tensor:


#     dev  = 'cuda'
#     dtype = global_points_onehot.dtype

#     fixed_point = fixed_point.to(device=dev, dtype=dtype)      # [2]
#     push_action = push_action.to(device=dev, dtype=dtype)      # [7]

#     xyz   = global_points_onehot[:, :3]                        # [N,3]
#     roles = global_points_onehot[:, 3:]                        # [N,3]  (保持不变)

#     t = push_action[:3]                                        # [tx, ty, tz]
#     q = push_action[3:7].unsqueeze(0)                          # [1,4]
#     R_push = _quat_to_rotmat_torch(q).squeeze(0)               # [3,3]

#     push_pose = torch.eye(4, device=dev, dtype=dtype)
#     push_pose[:3, :3] = R_push
#     push_pose[:3,  3] = t

#     tz = push_action[2]   
#     tz = tz.unsqueeze(0)                                   # [1]
#     trans_fixed = torch.cat([fixed_point, tz], dim=-1)         # [3] = [fx, fy, tz]

#     R_fixed = torch.tensor([[0., -1.,  0.],
#                             [-1.,  0.,  0.],
#                             [0.,   0., -1.]], device=dev, dtype=dtype)
#     fixed_pose = torch.eye(4, device=dev, dtype=dtype)
#     fixed_pose[:3, :3] = R_fixed
#     fixed_pose[:3,  3] = trans_fixed

#     T_2fixed = fixed_pose @ torch.linalg.inv(push_pose)        # [4,4]

#     N = xyz.shape[0]
#     ones = torch.ones((N, 1), device=dev, dtype=dtype)
#     xyz_h = torch.cat([xyz, ones], dim=1)                      # [N,4]
#     xyz_tf = (T_2fixed @ xyz_h.T).T[:, :3]                     # [N,3]

#     global_points_onehot_ee = torch.cat([xyz_tf, roles], dim=1)  # [N,6]

#     return global_points_onehot_ee

# def pc_normalize_for_obj_onehot(pc: torch.Tensor,
#                         centroid: torch.Tensor,
#                         m: torch.Tensor):
#     device, dtype = pc.device, pc.dtype
#     N, C = pc.shape

#     xyz   = pc[:, :3]                 # [N,3]
#     extra = pc[:, 3:] if C > 3 else None  # [N,C-3]（包含 one-hot 等）

#     xyz_c    = xyz - centroid         # 平移到原点

#     # 数值稳定：半径过小时用 1.0 避免除零/NaN
#     eps = torch.tensor(1e-12, device=device, dtype=dtype)
#     scale = torch.where(m > eps, m, torch.ones((), device=device, dtype=dtype))

#     xyz_n = xyz_c / scale             # 归一化

#     pc_norm = torch.cat([xyz_n, extra], dim=1) if extra is not None else xyz_n

#     return pc_norm

# def pc_normalize_grasp_onehot(pc: torch.Tensor):

#     if not torch.is_tensor(pc):
#         pc = torch.as_tensor(pc, dtype=torch.float32)
#     if pc.ndim != 2 or pc.shape[1] < 3:
#         raise ValueError(f"期望 [N,>=3]，收到 {tuple(pc.shape)}")

#     device, dtype = pc.device, pc.dtype
#     N, C = pc.shape

#     xyz   = pc[:, :3]                 # [N,3]
#     extra = pc[:, 3:] if C > 3 else None  # [N,C-3]（包含 one-hot 等）

#     centroid = xyz.mean(dim=0)        # [3]
#     xyz_c    = xyz - centroid         # 平移到原点

#     # 最大半径（最远点欧氏距离）
#     m = torch.linalg.norm(xyz_c, ord=2, dim=1).max()  # 标量张量
#     # 数值稳定：半径过小时用 1.0 避免除零/NaN
#     eps = torch.tensor(1e-12, device=device, dtype=dtype)
#     scale = torch.where(m > eps, m, torch.ones((), device=device, dtype=dtype))

#     xyz_n = xyz_c / scale             # 归一化

#     pc_norm = torch.cat([xyz_n, extra], dim=1) if extra is not None else xyz_n

#     return pc_norm, centroid, m

# def _quat_to_rotmat_torch(q: torch.Tensor) -> torch.Tensor:
#     """
#     q: (..., 4)  in [qx, qy, qz, qw]  (SciPy的 from_quat 默认顺序)
#     return: (..., 3, 3)
#     """
#     # 归一化
#     q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
#     qx, qy, qz, qw = q.unbind(-1)

#     # 按标准公式构造旋转矩阵
#     # 参考: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
#     xx, yy, zz = qx*qx, qy*qy, qz*qz
#     xy, xz, yz = qx*qy, qx*qz, qy*qz
#     wx, wy, wz = qw*qx, qw*qy, qw*qz

#     r00 = 1 - 2*(yy + zz)
#     r01 =     2*(xy - wz)
#     r02 =     2*(xz + wy)

#     r10 =     2*(xy + wz)
#     r11 = 1 - 2*(xx + zz)
#     r12 =     2*(yz - wx)

#     r20 =     2*(xz - wy)
#     r21 =     2*(yz + wx)
#     r22 = 1 - 2*(xx + yy)

#     R = torch.stack([
#         torch.stack([r00, r01, r02], dim=-1),
#         torch.stack([r10, r11, r12], dim=-1),
#         torch.stack([r20, r21, r22], dim=-1)
#     ], dim=-2)
#     return R

# def any_point_in_expanded_obb(
#     A: Union[o3d.geometry.PointCloud, np.ndarray],
#     B: Union[o3d.geometry.PointCloud, np.ndarray],
#     expand_by: float = 0.5
# ) -> bool:
#     """
#     判断点云 B 是否有点落在点云 A 的 OBB 放大后（中心不变）的包围盒内。
#     参数:
#         A: 点云A（Open3D PointCloud 或 (N,3) numpy 数组）
#         B: 点云B（Open3D PointCloud 或 (M,3) numpy 数组）
#         expand_by: 放大量，0.5 表示在各轴上扩大 50%（即 1.5×）
#     返回:
#         bool: 若 B 中至少一个点落入放大 OBB 内，返回 True，否则 False
#     """

#     # 1) 计算 A 的 OBB
#     obb = A.get_oriented_bounding_box()

#     # 2) 按各轴等比例放大 (中心不变)
#     scale = 1.0 + float(expand_by)
#     if scale <= 0:
#         raise ValueError("放大倍数无效：1 + expand_by 必须大于 0。")
#     obb_expanded = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, obb.extent.copy())
#     obb_expanded.scale(scale, obb_expanded.center)  # 原地缩放

#     # 3) 判断 B 是否有点落在放大后的 OBB 内
#     ptsB = np.asarray(B.points)
#     if ptsB.shape[0] == 0:
#         return False
#     ptsB_open3d = o3d.utility.Vector3dVector(ptsB)
#     inside_idx = obb_expanded.get_point_indices_within_bounding_box(ptsB_open3d)
#     return len(inside_idx) > 0

# def compute_distance_GOC(pose, obj_center):
#     """
#     Distance between the object's center of mass and the gripping plane under horizontal mapping.
#     Assuming the center is the center of mass.
#     """
#     p = np.asarray(pose, dtype=float)
#     c = np.asarray(obj_center, dtype=float)
#     diff_xy = (p - c)[..., :2]
#     return np.linalg.norm(diff_xy, axis=-1)

# # def compute_close_width(
# #     points_world: np.ndarray,
# #     grip_pos: np.ndarray,
# #     pad_width: float,                 
# #     clearance: float = 0.0,           
# #     centerline_y_offset: float = 0.0, 
# # ) -> float:
# #     """
# #     The final gripping width is determined 
# #     by adding a threshold to the critical 
# #     collision width between the gripper and the object being gripped.
# #     """
# #     P = TransformPCD2EndLink(points_world, grip_pos)  # (N,3)
# #     pose_points, _ = grasp_pcd_bluenoise_like(n_target=170, oversample=2000, seed=55926)
# #     sence_points = fuse_state_torch_v2(P, pose_points)
# #     sence_points = sence_points.cpu().numpy()
# #     # x, y, z = sence_points[:, 0], sence_points[:, 1], sence_points[:, 2]
# #     P = P.cpu().numpy()
# #     x, y, z = P[:, 0], P[:, 1], P[:, 2]

# #     # #----compute gripper mapping points (need y axis toward pad).
# #     half_pw = 0.5 * pad_width
# #     mask_band = (x >= -half_pw) & (x <= half_pw)
# #     y_band = y[mask_band] - centerline_y_offset
# #     #----no limit on width.
# #     if y_band.size == 0:
# #         return float("inf")

# #     #----find the farthest point.
# #     y_pos = y_band[y_band > 0.0]
# #     y_neg = y_band[y_band < 0.0]
# #     y_pos_max = np.max(y_pos) if y_pos.size > 0 else np.inf
# #     y_neg_min = np.min(y_neg) if y_neg.size > 0 else -np.inf

# #     #----compute the farthest distance on both sides.
# #     right_gap = y_pos_max if np.isfinite(y_pos_max) else np.inf
# #     left_gap  = -y_neg_min if np.isfinite(y_neg_min) else np.inf
# #     #----check if both sides is empty
# #     if not np.isfinite(right_gap) and not np.isfinite(left_gap):
# #         return float("inf")
# #     #----if one side is empty, the width depend on another one.
# #     gap_to_center = max(right_gap, left_gap) if np.isfinite(right_gap) and np.isfinite(left_gap) \
# #                     else (right_gap if np.isfinite(right_gap) else left_gap)
# #     #----set a clearance for safely grasp.
# #     half_width = max(0.0, gap_to_center + clearance)

# #     fig, ax = plt.subplots(figsize=(6, 6))

# #     # # 所有点（x,y）
# #     ax.scatter(x, y, s=2, alpha=0.2, label="all points")

# #     # # 带内点
# #     ax.scatter(x[mask_band], y[mask_band],
# #                 s=4, alpha=0.7, label="band points")

# #     # # pad 之间的 x 区间
# #     ax.axvline(-half_pw, color="black", linestyle="--", linewidth=1)
# #     ax.axvline( half_pw, color="black", linestyle="--", linewidth=1)
# #     ax.text(0, ax.get_ylim()[1]*0.95,
# #             f"pad_width = {pad_width:.3f}",
# #             ha="center", va="top")

# #     # # 中心线（考虑 y_offset）
# #     ax.axhline(centerline_y_offset, color="gray", linestyle=":", linewidth=1,
# #                 label="centerline + offset")

# #     # # 画出计算出的半宽（相对中心线）
# #     y_c = centerline_y_offset
# #     ax.axhline(y_c + half_width, color="red", linestyle="-", linewidth=1.5,
# #                 label=f"+half_width = {half_width:.3f}")
# #     ax.axhline(y_c - half_width, color="red", linestyle="-", linewidth=1.5)
# #     title = 'gripper width'
# #     width = 2.0 * half_width
# #     # # 把最远点标出来
# #     if np.isfinite(y_pos_max):
# #         ax.scatter(0, y_c + right_gap, color="green", s=40, marker="^",
# #                     label=f"right_gap = {right_gap:.3f}")
# #     if np.isfinite(left_gap):
# #         ax.scatter(0, y_c - left_gap, color="blue", s=40, marker="v",
# #                     label=f"left_gap = {left_gap:.3f}")

# #     ax.set_xlabel("x (along pad width)")
# #     ax.set_ylabel("y (between fingers)")
# #     ax.set_aspect("equal", adjustable="box")
# #     ax.legend(loc="best")
# #     ax.set_title(title or f"close width = {width:.3f}")

# #     plt.tight_layout()
# #     plt.show()
# #     return 2.0 * half_width

# def compute_close_width(
#     points_world: np.ndarray,
#     grip_pos: np.ndarray,
#     pad_width: float,                 
#     clearance: float = 0.0,           
#     centerline_y_offset: float = 0.0, 
#     ext = 0.0
# ) -> float:
#     """
#     根据点云计算最小夹爪开合（沿 y 方向），并可视化：
#     - pad 区域
#     - 物体点云
#     - 最大接触宽度
#     - 点云中心到夹爪中心线 x=0 的距离
#     """
#     P = TransformPCD2EndLink(points_world, grip_pos)  # (N,3), gripper 坐标系
#     pose_points, _ = grasp_pcd_bluenoise_like(extend=ext)
#     sence_points, _ = fuse_state_torch_v3(P, pose_points)
#     if len(sence_points) <= 50:
#         return 0, False
#     sence_points = sence_points.cpu().numpy()

#     # P = points_world.cpu().numpy()
#     x, y, z = sence_points[:, 0], sence_points[:, 1], sence_points[:, 2]

#     half_pw = 0.5 * pad_width
#     mask_band = (x >= -half_pw) & (x <= half_pw)

#     y_band = y[mask_band] - centerline_y_offset

#     if y_band.size == 0:
#         return float("inf"), False

#     y_pos = y_band[y_band > 0.0]
#     y_neg = y_band[y_band < 0.0]
#     y_pos_max = np.max(y_pos) if y_pos.size > 0 else np.inf
#     y_neg_min = np.min(y_neg) if y_neg.size > 0 else -np.inf

#     right_gap = y_pos_max if np.isfinite(y_pos_max) else np.inf
#     left_gap  = -y_neg_min if np.isfinite(y_neg_min) else np.inf

#     if not np.isfinite(right_gap) and not np.isfinite(left_gap):
#         return float("inf"), False

#     gap_to_center = (
#         max(right_gap, left_gap)
#         if np.isfinite(right_gap) and np.isfinite(left_gap)
#         else (right_gap if np.isfinite(right_gap) else left_gap)
#     )

#     half_width = max(0.0, gap_to_center + clearance)
#     width = 2.0 * half_width

#     # -------- 4) 可视化
#     # fig, ax = plt.subplots(figsize=(6, 6))

#     # # 所有点
#     # ax.scatter(x, y, s=2, alpha=0.2, label="all points")

#     # # 带内点
#     # ax.scatter(x_band, y_band_full, s=4, alpha=0.7, label="band points")

#     # # pad 边界（沿 x）
#     # ax.axvline(-half_pw, color="black", linestyle="--", linewidth=1)
#     # ax.axvline( half_pw, color="black", linestyle="--", linewidth=1)
#     # ax.text(0, ax.get_ylim()[1]*0.95,
#     #         f"pad_width = {pad_width:.3f}",
#     #         ha="center", va="top")

#     # # y 方向中心线
#     # y_c = centerline_y_offset
#     # ax.axhline(y_c, color="gray", linestyle=":", linewidth=1,
#     #            label="centerline + offset")

#     # # 开合半宽（沿 y）
#     # ax.axhline(y_c + half_width, color="red", linestyle="-", linewidth=1.5,
#     #            label=f"+half_width = {half_width:.3f}")
#     # ax.axhline(y_c - half_width, color="red", linestyle="-", linewidth=1.5)

#     # # -------- 5) 夹爪中心线 x = 0 以及点云中心到该线的距离
#     # # 夹爪中心线（过 (0,0)，垂直于 x 轴）
#     # ax.axvline(0.0, color="purple", linestyle="-.", linewidth=1.5,
#     #            label="gripper centerline (x=0)")

#     # # 点云中心点
#     # # ax.scatter(center_x, center_y, s=60, marker="*", color="purple",
#     # #            label=f"cloud center ({center_x:.3f}, {center_y:.3f})")

#     # # # 中心点到中心线的垂直距离线段（沿 x）
#     # # ax.plot([0.0, center_x], [center_y, center_y],
#     # #         color="purple", linewidth=2)

#     # # # 在中点位置标注距离数值
#     # # mid_x = 0.5 * center_x
#     # # ax.text(mid_x,
#     # #         center_y + 0.5 * (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
#     # #         f"d = {center_dist:.3f}",
#     # #         ha="center", va="bottom", color="purple")

#     # ax.set_xlabel("x (along pad width)")
#     # ax.set_ylabel("y (between fingers)")
#     # ax.set_aspect("equal", adjustable="box")
#     # ax.legend(loc="best")
#     # ax.set_title(f"gripper width = {width:.3f}")
#     # plt.tight_layout()
#     # plt.show()

#     # 如果后续还要用，可以把 center_dist 一并返回（例如改成 return width, center_dist）
#     return width, True


# def collision_dect(global_pc, obj_pc, grasp_pose):
#     """
#     Using the gripper pc to crop global pc to detect collison,
#     if no point in gripper boxs, we think there is no collison occur.
#     """

#     pass

# def convert_single_graspnet_grasp_to_env_pose(t_g: np.ndarray,
#                                             R_g: np.ndarray,
#                                             depth: float) -> np.ndarray:
#         """
#         将 GraspNet 的单个抓取 (t_g, R_g, depth) 转成环境里 grasp() 期望的 7D pose。

#         输入:
#             t_g   : (3,)  GraspNet 给出的 translation（抓取点，机械臂坐标系）
#             R_g   : (3,3) GraspNet 的旋转矩阵，列向量为 {x_g, y_g, z_g}，x_g 为接近方向
#             depth : float GraspNet 给出的深度 (沿 x_g 的距离)

#         输出:
#             grasp_pose: (7,) = [x, y, z, qx, qy, qz, qw]
#                         坐标系已经对齐到你当前 grasp() 使用的 tip 坐标系，
#                         可以直接传给 grasp(pose) 使用。
#         """

#         # ---------- 1) 把位置从“指尖接触点”移动到“抓取框架中心 / tip” ----------
#         # GraspNet 中 depth 通常定义为沿接近方向 x_g 到 gripper center 的距离
#         # t_tip = t_contact + depth * x_g
#         x_g = R_g[:, 0]                       # approach 方向
#         t_tip = t_g + depth * x_g             # tip 在机械臂坐标系下的位置

#         # ---------- 2) 旋转矩阵从 GraspNet 坐标系 G 映射到你环境的 tip 坐标系 ----------

#         R_tip = np.zeros((3, 3), dtype=np.float32)
#         R_tip[:, 0] = R_g[:, 2]       # x_tip 对齐 z_g
#         R_tip[:, 1] = -R_g[:, 1]      # y_tip 对齐 -y_g
#         R_tip[:, 2] = R_g[:, 0]       # z_tip 对齐 x_g (approach)

#         # ---------- 3) 修正左/右手系，确保 cross(x_tip, y_tip) ≈ z_tip ----------
#         if np.linalg.norm(np.cross(R_tip[:, 0], R_tip[:, 1]) - R_tip[:, 2]) > 0.1:
#             # 左手系: 把 x 轴取反，转成右手系
#             R_tip[:, 0] = -R_tip[:, 0]

#         # 再做一次正交化，稳一点
#         u, _, vh = np.linalg.svd(R_tip)
#         R_tip = u @ vh

#         grasp_pose = np.eye(4)
#         grasp_pose[:3, 3] = t_tip
#         grasp_pose[:3, :3] = R_tip

#         return grasp_pose

# def compute_approach_angle(approach_vector):
#     flag = True
#     x_axis = approach_vector
#     angle_rad = math.asin(abs(x_axis[2]) / np.linalg.norm(x_axis))
#     if math.degrees(angle_rad) < 45:
#         flag = False
#         # print(f"\033[32m ------------------------------------------ \033[0m")
#         # print(f"\033[32m approach angle = {math.degrees(angle_rad)} \033[0m")
#         # print(f"\033[32m ------------------------------------------ \033[0m")
#     return flag

# def adjust_approach_angle(R: np.ndarray,
#                           min_angle_deg: float = 30.0,
#                           axis_index: int = 0) -> np.ndarray:
#     """
#     调整旋转矩阵中 approach 轴与水平面的夹角：
#     - 若当前角度 < min_angle_deg，则把它调到恰好 min_angle_deg；
#     - 若 >= min_angle_deg，则保持不变。
    
#     R         : (3,3) 旋转矩阵
#     min_angle_deg : 与 XY 平面的最小夹角（单位：度）
#     axis_index    : 哪一列是 approach 轴（0/1/2），你现在是 [:3,0] 就用 0。
#     """
#     R = np.asarray(R, dtype=float).reshape(3, 3)

#     # 取出当前 approach 轴并单位化
#     v = R[:, axis_index]
#     v = v / (np.linalg.norm(v) + 1e-12)

#     # 当前与 XY 平面的夹角（0° = 完全水平, 90° = 垂直）
#     angle_rad = math.asin(abs(v[2]) / (np.linalg.norm(v) + 1e-12))
#     angle_deg = math.degrees(angle_rad)
#     print(f"\033[32m approach angle (before) = {angle_deg:.3f} deg \033[0m")

#     if angle_deg >= min_angle_deg - 1e-6:
#         # 已满足要求，直接返回
#         return R.copy()

#     # ------------ 构造与平面夹角 = min_angle_deg 的目标向量 v_new ------------
#     target_rad = math.radians(min_angle_deg)

#     # 保持 XY 投影方向不变
#     v_xy = np.array([v[0], v[1], 0.0], dtype=float)
#     norm_xy = np.linalg.norm(v_xy)
#     if norm_xy < 1e-8:
#         # 几乎垂直，理论上不会进入 angle < min_angle 的分支，保险起见原样返回
#         return R.copy()

#     dir_xy = v_xy / norm_xy                 # XY 平面中的方向
#     v_new_xy = math.cos(target_rad) * dir_xy
#     sign_z = 1.0 if v[2] >= 0.0 else -1.0   # 保持原来的“向上/向下”符号
#     v_new_z = sign_z * math.sin(target_rad)

#     v_new = np.array([v_new_xy[0], v_new_xy[1], v_new_z], dtype=float)
#     v_new /= (np.linalg.norm(v_new) + 1e-12)

#     # ------------ 计算把 v 旋到 v_new 的最小旋转 ΔR（Rodrigues）------------
#     dot = float(np.clip(np.dot(v, v_new), -1.0, 1.0))
#     if abs(dot - 1.0) < 1e-8:
#         # 几乎已对齐
#         return R.copy()

#     angle = math.acos(dot)
#     axis = np.cross(v, v_new)
#     axis_norm = np.linalg.norm(axis)
#     if axis_norm < 1e-8:
#         # 数值退化，直接返回
#         return R.copy()
#     axis = axis / axis_norm

#     x, y, z = axis
#     K = np.array([[0, -z,  y],
#                   [z,  0, -x],
#                   [-y, x,  0]])
#     dR = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)

#     # 对整个姿态做刚体旋转
#     R_new = dR @ R

#     # 检查新角度
#     v_chk = R_new[:, axis_index]
#     angle_new = math.degrees(
#         math.asin(abs(v_chk[2]) / (np.linalg.norm(v_chk) + 1e-12))
#     )
#     print(f"\033[32m approach angle (after)  = {angle_new:.3f} deg \033[0m")
#     print(f"\033[32m --------------------------------------------- \033[0m")
#     return R_new

# def compute_location_score(pose_translation, pcd, distance_force):
    
#     obb, center, max_vertical_distance = analyze_point_cloud(pcd)
#     pose_translation = np.array(pose_translation)
#     center = np.array(center)
#     distance_force = np.linalg.norm(pose_translation[:2] - center[:2])
#     score = 1 - distance_force / max_vertical_distance
#     # --- 可视化 ---
#     # 创建一个坐标球代表质心
#     center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
#     center_sphere.paint_uniform_color([0, 0, 1]) # 蓝色
#     center_sphere.translate(center)

#     # 创建坐标轴
#     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=center)

#     print("正在打开可视化窗口...")
#     o3d.visualization.draw_geometries(
#         [pcd, obb, center_sphere, mesh_frame], 
#         window_name="包围框与质心分析",
#         width=800, height=600
#     )
#     return score

# def analyze_point_cloud(pcd):
#     # ---------------------------------------------------------
#     # 1. 估计点云中心 (质心)
#     # ---------------------------------------------------------
#     # get_center() 返回所有点的算术平均值
#     center = pcd.get_center()
#     print(f"1. 点云质心 (Center): {center}")
#     # ---------------------------------------------------------
#     # 4. (可选) 生成定向包围框 (OBB - Oriented Bounding Box)
#     # ---------------------------------------------------------
#     # OBB 是最小体积包围框，可以任意旋转
#     obb = pcd.get_oriented_bounding_box()
#     obb.color = (0, 1, 0)  # 绿色显示 OBB
#     # OBB 的"垂直"是相对于物体自身的局部坐标系的
#     # extent 返回 [width, height, depth]
#     obb_extent = obb.extent
#     max_length = max(obb_extent)
#     print(f"6. OBB 局部尺寸 (Extent): {obb_extent}")
#     print(f"   max_force_length: {max_length / 2:.4f}")

#     return obb, center, max_length / 2

# def visualize_center_distance_obb(points_world, grip_pos):
#     """
#     1) 将点云变换到末端坐标系：P = TransformPCD2EndLink(points_world, grip_pos)
#     2) 用 Open3D 计算 P 的 OBB
#     3) 取 OBB 中心点作为“点云中心”，计算并可视化：
#          - D: OBB 中心到夹爪中心线 x=0 在 xy 平面的距离
#          - d: OBB 中心到 OBB 某一对平面之一的最大垂直距离
#             （沿最长轴方向的 1/2 extent）
#     4) 在 xy 平面上画出：
#          - 全部点云
#          - 夹爪中心线 x=0
#          - OBB 中心点
#          - D 的线段
#          - d 在 xy 平面的投影线段
#     返回:
#         D: float, OBB 中心到 x=0 直线在 xy 平面的距离
#         center: np.ndarray, (3,), OBB 中心
#         d_max_plane: float, OBB 中心到 OBB 框平面的最大垂直距离
#     """
#     # -------- 1) 变换到末端坐标系
#     P = TransformPCD2EndLink(points_world, grip_pos)  # (N, 3)

#     # 若是 torch.Tensor，转为 numpy
#     if hasattr(P, "cpu"):
#         P = P.cpu().numpy()
#     else:
#         P = np.asarray(P)

#     x, y, z = P[:, 0], P[:, 1], P[:, 2]

#     # -------- 2) 计算 OBB
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(P)

#     obb = pcd.get_oriented_bounding_box()
#     center = np.asarray(obb.center)       # (3,)
#     extent = np.asarray(obb.extent)       # (3,)  [ex, ey, ez]
#     R = np.asarray(obb.R)                 # (3, 3)  每列为一个主轴方向

#     cx, cy, cz = center

#     # 夹爪中心线：过 (0,0)，垂直 X 轴 → x = 0
#     D = float(abs(cx))   # 在 xy 平面的距离

#     # -------- 3) 计算 d：中心到 OBB 某一对平面之一的最大垂直距离
#     # 最长轴
#     i_max = int(extent.argmax())
#     # 该轴方向（单位向量）
#     axis_dir = R[:, i_max]
#     axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)

#     # 中心到该轴对应平面的距离
#     d_max_plane = 0.5 * float(extent[i_max])

#     # 中心指向平面上的端点（3D）
#     end_point_3d = center + axis_dir * d_max_plane  # (3,)

#     # 在 xy 平面的投影
#     end_x, end_y = end_point_3d[0], end_point_3d[1]

#     # -------- 4) 可视化（xy 平面）
#     fig, ax = plt.subplots(figsize=(6, 6))

#     # 所有点
#     ax.scatter(x, y, s=2, alpha=0.3, label="points")

#     # 夹爪中心线 x = 0 （计算 D 用）
#     ax.axvline(0.0, color="black", linestyle="--", linewidth=1,
#                label="gripper centerline (x=0)")

#     # OBB 中心点
#     ax.scatter(cx, cy, s=60, marker="*", color="red",
#                label=f"OBB center ({cx:.3f}, {cy:.3f})")

#     # D：中心到 x=0 的线段
#     ax.plot([0.0, cx], [cy, cy], color="red", linewidth=2)
#     mid_x_D = 0.5 * cx
#     ax.text(mid_x_D, cy,
#             f"D = {D:.3f}",
#             ha="center", va="bottom", color="red")

#     # d：中心到 OBB 最远平面的距离线段（在 xy 平面的投影）
#     ax.plot([cx, end_x], [cy, end_y],
#             color="orange", linewidth=2,
#             label=f"d (to OBB face) = {d_max_plane:.3f}")

#     # d 的文字标注放在线段中点
#     mid_x_d = 0.5 * (cx + end_x)
#     mid_y_d = 0.5 * (cy + end_y)
#     ax.text(mid_x_d, mid_y_d,
#             f"d = {d_max_plane:.3f}",
#             ha="center", va="bottom", color="orange")

#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_aspect("equal", adjustable="box")
#     ax.legend(loc="best")
#     ax.set_title(
#         f"OBB center distance: D={D:.3f} (to x=0), "
#         f"d={d_max_plane:.3f} (to OBB face)"
#     )
#     plt.tight_layout()
#     plt.show()
#     print(f'grasp score:{1 - D/d_max_plane}')
#     return D, center, d_max_plane

# def check_collision_voxel_then_kdtree(global_points_ee_tensor: torch.Tensor,
#                                       pose_points_tensor: torch.Tensor,
#                                       voxel_size_vox: float = 0.005,
#                                       threshold: float = 0.003) -> bool:
#     vox_env = _voxel_keys_from_tensor(global_points_ee_tensor, voxel_size_vox)
#     vox_pose = _voxel_keys_from_tensor(pose_points_tensor, voxel_size_vox)
#     vox_pose_dilated = _dilate_voxel_keys(vox_pose)

#     if len(vox_env) == 0 or len(vox_pose_dilated) == 0 or len(vox_env.intersection(vox_pose_dilated)) == 0:
#         return False

#     env_np = global_points_ee_tensor.detach().cpu().numpy()
#     pose_np = pose_points_tensor.detach().cpu().numpy()
#     if env_np.shape[0] == 0 or pose_np.shape[0] == 0:
#         return False

#     tree = cKDTree(env_np)
#     hits = tree.query_ball_point(pose_np, r=threshold)
#     return any(len(h) > 0 for h in hits)

# def _voxel_keys_from_tensor(points_tensor: torch.Tensor, voxel_size: float):
#     if points_tensor.numel() == 0:
#         return set()
#     pts = points_tensor.detach().cpu().numpy()
#     idx = np.floor(pts / voxel_size).astype(np.int64)
#     return set(map(tuple, idx))

# def _dilate_voxel_keys(voxel_keys: set):
#     offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
#     out = set()
#     for k in voxel_keys:
#         i, j, kz = k
#         for dx, dy, dz in offsets:
#             out.add((i+dx, j+dy, kz+dz))
#     return out

# def refine_pose(pose, depth_i, rotation_j, depth_bin=0.008, rotation_bin=15):
#     """
#     refine pose according to depth bin and rotation bin.
#     depth bin = 0.005 (1~3)
#     rotation bin = 5 (-15~15)
#     """
#     pose = np.asarray(pose, dtype=np.float64)
#     if pose.shape != (4, 4):
#         raise ValueError(f"pose must be a 4x4 homogeneous matrix, got {pose.shape}")

#     depth_refine = float(depth_i) * float(depth_bin)                 
                                
#     Tz = np.eye(4, dtype=np.float64)
#     R = np.eye(4, dtype=np.float64)
#     Tz[2, 3] = depth_refine   
#     if rotation_j != 0:
#         if rotation_j % 2 == 1:
#             rotation_j = (rotation_j + 1) / 2
#             rot_deg = float(rotation_j) * float(rotation_bin) # 1 3 5
#         else:
#             rotation_j = -rotation_j / 2
#             rot_deg = float(rotation_j) * float(rotation_bin)    
#         theta = np.deg2rad(rot_deg) 
#         c, s = np.cos(theta), np.sin(theta)
#         Rz = np.array([[ c, -s, 0.0],
#                     [ s,  c, 0.0],
#                     [0.0, 0.0, 1.0]], dtype=np.float64)
#         R[:3, :3] = Rz

#     delta = Tz @ R
#     refined_pose = pose @ delta
#     return refined_pose, int(rotation_j)

# def refine_pose_v2(pose, forward_j, depth_bin=0.01, forward_bin=0.015):
#     """
#     depth bin = 0.005 
#     forward_bin = 0.005
#     """
#     pose = np.asarray(pose, dtype=np.float64)
#     if pose.shape != (4, 4):
#         raise ValueError(f"pose must be a 4x4 homogeneous matrix, got {pose.shape}")
             
#     Tz = np.eye(4, dtype=np.float64)
#     Tx = np.eye(4, dtype=np.float64)
#     # if forward_j == 0:
#     depth_i = 1
#     depth_refine = float(depth_i) * float(depth_bin)   
#     Tz[2, 3] = depth_refine   
#     if forward_j != 0:
#         if forward_j % 2 == 1:
#             forward_j = (forward_j + 1) / 2
#             forward_deg = float(forward_j) * float(forward_bin) # 1 -1
#             Tx[0, 3] = forward_deg
#         else:
#             forward_j = -forward_j / 2
#             forward_deg = float(forward_j) * float(forward_bin)    
#             Tx[0, 3] = forward_deg
#     delta = Tz @ Tx
#     refined_pose = pose @ delta
#     return refined_pose, int(forward_j)

# def crop_cloud_by_pose_matrix(
#     points, 
#     radius=0.10, 
#     height=0.0475, 
#     angle_deg=30.0,
#     pose_matrix=None
# ):
#     """
#     输入:
#         points: (N, 3) 原始点云
#         pose_matrix: (4, 4) 齐次变换矩阵。
#                      - Col 0 (X轴): 圆柱轴线 (顶面指向底面)
#                      - Col 1 (Y轴): 平行于切线的直径
#                      - Col 2 (Z轴): 垂直于切线的方向
#                      - Col 3 (Origin): 顶面中心
#         radius, height: 几何尺寸
#         angle_deg: 切角
#     输出:
#         inside_points: 内部点云
#         mask: 布尔掩码
#     """
    
#     # --- 1. 将全局点云转换到局部坐标系 ---
#     # 局部坐标系定义: 原点在顶面中心，X朝底面，Y平行切线，Z垂直切线
    
#     # 方法 A: 使用矩阵逆 (数学上最严谨)
#     # T_local_to_global = pose_matrix
#     # T_global_to_local = inv(pose_matrix)
#     # Point_local = T_global_to_local * Point_global
    
#     # 为了计算高效，我们可以手动提取旋转和平移 (利用旋转矩阵的正交性 R^-1 = R^T)
#     pose_matrix = np.eye(4)
#     R = pose_matrix[:3, :3]
#     t = pose_matrix[:3, 3]
    
#     # 向量 P_vec = P_global - Origin
#     diff_vec = points - t
    
#     local_x = np.dot(diff_vec, R[:, 0]) # 沿 X轴 (高度方向) 
#     local_y = np.dot(diff_vec, R[:, 1]) # 沿 Y轴 
#     local_z = np.dot(diff_vec, R[:, 2]) # 沿 Z轴 

#     mask_height = (local_x >= -height) & (local_x <= 0)

#     mask_radius = (local_y**2 + local_z**2) <= radius**2 

#     cut_half_width = radius * np.sin(np.deg2rad(angle_deg)) 
#     mask_cut = np.abs(local_z) <= cut_half_width 

#     final_mask = mask_height & mask_radius & mask_cut 
#     return points[final_mask], final_mask

# def visualize_result(points, mask, pose_matrix):
#     """
#     可视化函数: 显示世界坐标系、输入位姿坐标系、内部点(红)、外部点(灰)
#     """
#     print(f"总点数: {len(points)}")
#     print(f"内部点数 (红色): {np.sum(mask)}")
    
#     # 1. 内部点云 (红色)
#     pcd_in = o3d.geometry.PointCloud()
#     pcd_in.points = o3d.utility.Vector3dVector(points[mask])
#     pcd_in.paint_uniform_color([1, 0, 0]) # Red
    
#     # 2. 外部点云 (灰色, 半透明感)
#     pcd_out = o3d.geometry.PointCloud()
#     pcd_out.points = o3d.utility.Vector3dVector(points[~mask])
#     pcd_out.paint_uniform_color([0.8, 0.8, 0.8]) # Grey
    
#     # 3. 世界坐标系 (原点)
#     frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0,0,0])
    
#     # 4. 输入位姿坐标系 (Pose Frame)
#     # Open3D 的 create_coordinate_frame 默认是在原点，RGB对应XYZ
#     # 直接应用 transform(pose_matrix) 即可把它移动到指定位姿
#     frame_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
#     frame_pose.transform(pose_matrix)
    
#     print("可视化说明:")
#     print("  - [大坐标轴] 世界原点")
#     print("  - [悬浮坐标轴] 输入位姿:")
#     print("      -> 红色 (X): 圆柱高度方向")
#     print("      -> 绿色 (Y): 切线平行方向")
#     print("      -> 蓝色 (Z): 切面宽度控制方向")
    
#     o3d.visualization.draw_geometries(
#         [pcd_out, pcd_in, frame_world, frame_pose],
#         window_name="Matrix Pose Crop Check",
#         width=1024, height=768
#     )



# def get_mask_bottom_origin_z_axis(local_points, radius, height, extend=0, angle_deg=30.0):
#     """
#     输入: 已经变换到局部坐标系的点云 (N, 3)
    
#     坐标系严格定义:
#       - Origin (0,0,0): 底面中心 (Bottom Center)
#       - Z axis (+Z): 顶面 -> 底面 (Top -> Bottom)
#         -> 推导: 底面是0，顶面是-Height。圆柱体位于 Z轴的 [-Height, 0] 区间。
#       - Y axis: 平行于切线
#       - X axis: 右手系生成的切面法线 (控制宽度)
#     """
#     x = local_points[:, 0]
#     y = local_points[:, 1]
#     z = local_points[:, 2]
    
#     # --- 1. 高度约束 (Z轴) ---
#     # 关键修正点: 有效区间是 [-Height, 0]
#     mask_height = (z >= -height) & (z <= extend)
    
#     # --- 2. 切面宽度约束 (X轴) ---
#     # X轴是切面的法线，控制保留宽度
#     cut_half_width = radius * np.sin(np.deg2rad(angle_deg))
#     mask_cut = np.abs(x) <= cut_half_width
    
#     # --- 3. 圆柱半径约束 (XY平面) ---
#     # 截面是垂直于 Z轴 的 XY 平面
#     mask_radius = (x**2 + y**2) <= radius**2
    
#     # --- 4. 综合 ---
#     final_mask = mask_height & mask_cut & mask_radius
    
#     return final_mask, cut_half_width

# def visualize_correction(local_points, radius=0.06, height=0.1,extend=0.008, angle_deg=15.0):
#     # 1. 计算 Mask
#     mask, cut_w = get_mask_bottom_origin_z_axis(local_points, radius, height, extend, angle_deg)
    
#     print(f"输入点数: {len(local_points)}")
#     print(f"内部点数: {np.sum(mask)}")
    
#     # 2. 颜色区分
#     pcd_in = o3d.geometry.PointCloud()
#     pcd_in.points = o3d.utility.Vector3dVector(local_points[mask])
#     pcd_in.paint_uniform_color([1, 0, 0]) # 红色 (内部)
    
#     pcd_out = o3d.geometry.PointCloud()
#     pcd_out.points = o3d.utility.Vector3dVector(local_points[~mask])
#     pcd_out.paint_uniform_color([0.9, 0.9, 0.9]) # 灰色 (外部)
    
#     # 3. 坐标轴 (原点 = 底面中心)
#     # 红色=X, 绿色=Y, 蓝色=Z
#     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=radius, origin=[0,0,0])
    
#     # 4. 绘制蓝色线框 (辅助验证)
#     # 这里的关键是 Z 的坐标范围是 [-height, 0]
    
#     # 定义8个顶点
#     # Z=0 (底面), Z=-height (顶面)
#     # X= +/- cut_w
#     # Y= +/- radius (为了画框方便，画个外接矩形)
    
#     box_corners = [
#         [-cut_w, -radius, 0],       [-cut_w, radius, 0],       # Bottom (-X)
#         [cut_w, -radius, 0],        [cut_w, radius, 0],        # Bottom (+X)
#         [-cut_w, -radius, -height], [-cut_w, radius, -height], # Top (-X)
#         [cut_w, -radius, -height],  [cut_w, radius, -height]   # Top (+X)
#     ]
    
#     # 连接线
#     lines = [
#         [0, 1], [1, 3], [3, 2], [2, 0], # 底面框 (Z=0)
#         [4, 5], [5, 7], [7, 6], [6, 4], # 顶面框 (Z=-height)
#         [0, 4], [1, 5], [2, 6], [3, 7]  # 连接棱
#     ]
    
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(box_corners)
#     line_set.lines = o3d.utility.Vector2iVector(lines)
#     line_set.paint_uniform_color([0, 0, 1]) # 蓝色
    
#     print("\n可视化验证 (Bottom Origin + Z Axis Corrected):")
#     print("1. [坐标原点]: 位于底面中心。")
#     print("2. [蓝色轴 Z]: 指向 Top->Bottom 方向。")
#     print("   -> 注意: 红色点云应该位于蓝色轴的**反方向** (负半轴)。")
#     print("   -> 也就是 Z轴 像尾巴一样从红色点云的屁股后面伸出来。")
#     print("3. [蓝色线框]: 完美包裹住红色点云，且 Z 范围是 [0 到 -H]。")
    
#     o3d.visualization.draw_geometries(
#         [pcd_in, pcd_out, frame, line_set],
#         window_name="Corrected: Bottom Origin, Z-Axis",
#         width=1024, height=768
#     )


# # def get_mask_bottom_origin_z_axis_torch(
# #     local_points: torch.Tensor,   # [N,3] torch
# #     radius: float,
# #     height: float,
# #     extend: float = 0.0,
# #     angle_deg: float = 30.0,
# # ):
# #     """
# #     输入: 已经变换到局部坐标系的点云 local_points (N,3), torch tensor
# #     输出: mask (N,) torch.bool, cut_half_width float
# #     """
# #     x = local_points[:, 0]
# #     y = local_points[:, 1]
# #     z = local_points[:, 2]

# #     # Z: [-height, extend]
# #     mask_height = (z >= -height) & (z <= extend)

# #     # X cut
# #     cut_half_width = radius * math.sin(math.radians(angle_deg))
# #     mask_cut = x.abs() <= cut_half_width

# #     # radius in XY
# #     r2 = radius * radius
# #     mask_radius = (x * x + y * y) <= r2

# #     return (mask_height & mask_cut & mask_radius), cut_half_width

# def get_mask_bottom_origin_z_axis_torch(
#     local_points: torch.Tensor,
#     radius: float,
#     height: float,
#     extend: float = 0.0,
#     angle_deg: float = 30.0,
# ):
#     x = local_points[:, 0]
#     y = local_points[:, 1]
#     z = local_points[:, 2]

#     sin_theta = math.sin(math.radians(angle_deg))
#     cut_half_width = radius * sin_theta
#     cut_half_width2 = cut_half_width * cut_half_width
#     r2 = radius * radius

#     x2 = x * x
#     mask = (
#         (z >= -height) &
#         (z <= extend) &
#         (x2 <= cut_half_width2) &
#         ((x2 + y * y) <= r2)
#     )
#     return mask, cut_half_width


# from functools import lru_cache

# # 你已有的 gripper_point_width(n_target=200, oversample=..., gripper_width=...) 继续用
# # 这里仅做缓存 + 降低 oversample（你可按精度需求调 600~1200）

# @lru_cache(maxsize=512)
# def _gripper_points_cached(width_mm: int):
#     w = width_mm / 1000.0
#     pts, _ = gripper_point_width(n_target=200, oversample=5000, gripper_width=w)  # oversample 下调
#     # pts 通常在 CPU，缓存为 CPU tensor，使用时再搬到 GPU
#     return pts.contiguous()

# def get_gripper_points(width: float, device: torch.device, dtype: torch.dtype):
#     width_mm = int(round(float(width) * 1000.0))
#     pts = _gripper_points_cached(width_mm)
#     return pts.to(device=device, dtype=dtype, non_blocking=True)



import math
import numpy as np
import pybullet as p
import cv2
import open3d as o3d
import re
from shapely.geometry import Point, Polygon,MultiPolygon
from scipy.ndimage import binary_fill_holes
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
# from refine.env.constants import WORKSPACE_LIMITS, PIXEL_SIZE
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from typing import Union

from typing import Literal, Optional, Tuple, Union
try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

reconstruction_config = {
    'nb_neighbors': 50,
    'std_ratio': 2.0,
    'voxel_size': 0.0015,
    'icp_max_try': 5,
    'icp_max_iter': 2000,
    'translation_thresh': 3.95,
    'rotation_thresh': 0.02,
    'max_correspondence_distance': 0.02
}

graspnet_config = {
    'graspnet_checkpoint_path': '/home/ubuntu/task/more_than_grasp/models/graspnet/checkpoints/checkpoint-rs.tar',
    'refine_approach_dist': 0.01,
    'dist_thresh': 0.05,
    'angle_thresh': 10,
    'mask_thresh': 0.5
}
# graspnet_config = {
#     'graspnet_checkpoint_path': '/home/ubuntu/task/MyProject_Grasp-Push/models/graspnet/checkpoints/checkpoint-rs.tar',
#     'refine_approach_dist': 0.015,
#     'dist_thresh': 0.05,
#     'angle_thresh': 4,
#     'mask_thresh': 0.3
# }
def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.
    Args:
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
        points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)

    return points

def get_all_pointcloud(depth, intrinsics, seg, target_id):

    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)

    segm = np.array(seg, dtype=np.int32).reshape(depth.shape)
    object_mask = (segm == target_id)|(segm == 1)
    object_pcd_mask = (segm == target_id)

    depth_mask = depth.copy()
    depth_mask[~object_mask] = 0  

    depth_pcd = depth.copy()
    depth_pcd[~object_pcd_mask] = 0

    # px, py = np.meshgrid(xlin, ylin)
    # px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    # py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points2 = np.float32([px, py, depth_mask]).transpose(1, 2, 0)
    points_pcd = np.float32([px, py, depth_pcd]).transpose(1, 2, 0)

    return points, points2, points_pcd

def get_mask_pointcloud(depth, intrinsics, seg, target_id):
    """

    """
    segm = np.array(seg, dtype=np.int32).reshape(depth.shape)
    object_mask = (segm == target_id)|(segm == 1)
    object_pcd_mask = (segm == target_id)
    
    # segm = np.array(seg, dtype=np.int32).reshape(depth.shape)

    # # segmentationMaskBuffer = objectUniqueId + (linkIndex+1)<<24
    # obj_ids   = segm & ((1 << 24) - 1)    # 取低 24 bit => objectUniqueId
    # # 根据 objectUniqueId 构造 mask（假设 target_id 是 bodyUniqueId）
    # object_mask     = (obj_ids == target_id) | (obj_ids == 1)  # 1 可以是平面 / 桌子等
    # object_pcd_mask = (obj_ids == target_id)

    depth_mask = depth.copy()
    depth_mask[~object_mask] = 0  

    depth_pcd = depth.copy()
    depth_pcd[~object_pcd_mask] = 0

    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth_mask]).transpose(1, 2, 0)
    points_pcd = np.float32([px, py, depth_pcd]).transpose(1, 2, 0)
    return points, points_pcd

def get_all_obj_mask_pointcloud(depth, intrinsics, seg, all_obj_id):
    """
    get all obj points for push sample
    """
    segm = np.array(seg, dtype=np.int32).reshape(depth.shape)
    all_points = []

    for id in all_obj_id:
        object_pcd_mask = (segm == id)

        depth_pcd = depth.copy()
        depth_pcd[~object_pcd_mask] = 0

        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
        py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])

        point_pcd = np.float32([px, py, depth_pcd]).transpose(1, 2, 0)
        all_points.append(point_pcd)
    return all_points

def get_obj_pointcloud(depth, intrinsics, seg, target_id):
    """

    """
    segm = np.array(seg, dtype=np.int32).reshape(depth.shape)
    object_mask = (segm == target_id)
    
    depth = depth.copy()
    depth[~object_mask] = 0  
    
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)

    return points 

def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
        points: HxWx3 float array of 3D points in camera coordinates.
        transform: 4x4 float array representing a rigid transformation matrix.
    Returns:
        points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding, "constant", constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points

def process_pcds(pcds, reconstruction_config):
    trans = dict()
    pcd = pcds[0]
    pcd.estimate_normals()
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors = reconstruction_config['nb_neighbors'],
        std_ratio = reconstruction_config['std_ratio']
    )
    for i in range(1, len(pcds)):
        voxel_size = reconstruction_config['voxel_size']
        income_pcd, _ = pcds[i].remove_statistical_outlier(
            nb_neighbors = reconstruction_config['nb_neighbors'],
            std_ratio = reconstruction_config['std_ratio']
        )
        income_pcd.estimate_normals()
        income_pcd = income_pcd.voxel_down_sample(voxel_size)
        transok_flag = False
        for _ in range(reconstruction_config['icp_max_try']): # try 5 times max
            reg_p2p = o3d.pipelines.registration.registration_icp(
                income_pcd,
                pcd,
                reconstruction_config['max_correspondence_distance'],
                np.eye(4, dtype = np.float),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(reconstruction_config['icp_max_iter'])
            )
            if (np.trace(reg_p2p.transformation) > reconstruction_config['translation_thresh']) \
                and (np.linalg.norm(reg_p2p.transformation[:3, 3]) < reconstruction_config['rotation_thresh']):
                # trace for transformation matrix should be larger than 3.5
                # translation should less than 0.05
                transok_flag = True
                break
        if not transok_flag:
            reg_p2p.transformation = np.eye(4, dtype = np.float32)
        income_pcd = income_pcd.transform(reg_p2p.transformation)
        trans[i] = reg_p2p.transformation
        pcd += income_pcd
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals()
    return trans, pcd

def process_pcds_test(pcds):
    points_state_list = []
    colors = []
    for pcd in pcds:
        points = np.asarray(pcd.points)
        color = np.asarray(pcd.colors)
        points_state_list.append(points)
        colors.append(color)

    points_state = np.vstack(points_state_list)
    colors_state = np.vstack(colors)
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(points_state)
    points_pcd.colors = o3d.utility.Vector3dVector(colors_state)

    return points_pcd

def process_all_pcds(all_config_pcd):
    obj_points_list = []

    for i in range(len(all_config_pcd[0])):
        obj_point = []
        for j in range(len(all_config_pcd)):
            pcd = all_config_pcd[j][i]
            points = np.asarray(pcd.points)

            obj_point.append(points)

        obj_point = np.vstack(obj_point)
        obj_points_list.append(obj_point)
    obj_pcds_list = []
    for i in range(len(obj_points_list)):
        points_pcd = o3d.geometry.PointCloud()
        points_pcd.points = o3d.utility.Vector3dVector(obj_points_list[i])
        obj_pcds_list.append(points_pcd)

    return obj_pcds_list

def get_fuse_pointcloud(env, obj_id, id=0):
    pcds = []
    configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        color, depth, seg = env.render_camera(config)
        if id == 0:
            xyz, _ = get_mask_pointcloud(depth, config["intrinsics"], seg, obj_id)
        else:
            _, xyz = get_mask_pointcloud(depth, config["intrinsics"], seg, obj_id)
        # xyz = get_pointcloud(depth, config["intrinsics"])
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        points = transform_pointcloud(xyz, transform)
        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
        iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
        iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = color[valid]
        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        # pcd.voxel_down_sample(reconstruction_config['voxel_size'])
        # # visualization
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        # the first pcd is the one for start fusion
        pcds.append(pcd)

    # _, fuse_pcd = process_pcds(pcds, reconstruction_config)
    fuse_pcd = process_pcds_test(pcds)
    # visualization
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([fuse_pcd, frame])

    return fuse_pcd

def get_all_obj_pointcloud(env, obj_lis):
    all_config_pcds = []
    configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        one_config_pcds = []
        color, depth, seg = env.render_camera(config)
        all_xyz = get_all_obj_mask_pointcloud(depth, config['intrinsics'], seg, obj_lis)
        # xyz = get_pointcloud(depth, config["intrinsics"])
        for xyz in all_xyz:
            position = np.array(config["position"]).reshape(3, 1)
            rotation = p.getMatrixFromQuaternion(config["rotation"])
            rotation = np.array(rotation).reshape(3, 3)
            transform = np.eye(4)
            transform[:3, :] = np.hstack((rotation, position))
            points = transform_pointcloud(xyz, transform)
            # Filter out 3D points that are outside of the predefined bounds.
            ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
            iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
            iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
            valid = ix & iy & iz
            points = points[valid]
            colors = color[valid]
            # Sort 3D points by z-value, which works with array assignment to simulate
            # z-buffering for rendering the heightmap image.
            iz = np.argsort(points[:, -1])
            points, colors = points[iz], colors[iz]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            pcd.voxel_down_sample(reconstruction_config['voxel_size'])
            # # visualization
            # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            # o3d.visualization.draw_geometries([pcd, frame])
            # the first pcd is the one for start fusion
            one_config_pcds.append(pcd)
        all_config_pcds.append(one_config_pcds)
    # _, fuse_pcd = process_pcds(pcds, reconstruction_config)
    fuse_pcd = process_all_pcds(all_config_pcds)
    # visualization
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([fuse_pcd, frame])

    return fuse_pcd


def global_label_points(depth, intrinsics, seg, target_id):
    """
    assign labels to goal-obj and other obj
    goal-obj:[0,1,0] other:[0,0,1]
    """
    segm = np.array(seg, dtype=np.int32).reshape(depth.shape)
    goal_obj_mask = (segm == target_id)
    other_mask = (segm != target_id) & (segm != 1)  
    without_floor = (segm == 1)
    # crop floor
    depth_mask_global = depth.copy()
    depth_mask_global[without_floor] = 0
    # depth_mask_obj = depth.copy()
    # depth_mask_obj[~goal_obj_mask] = 0
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    
    points = np.float32([px, py, depth_mask_global]).transpose(1, 2, 0)
    # obj_points = np.float32([px, py, depth_mask_obj]).transpose(1, 2, 0)
    # add labels to global_pc
    labels = np.zeros((height, width, 2), dtype=np.float32)
    labels[goal_obj_mask] = [1.0, 0.0]  # goal-obj
    labels[other_mask] = [0.0, 1.0] # other-obj
    global_points_six = np.concatenate([points, labels], axis=-1) 

    return global_points_six

def get_global_pc(env):
    pcds = []
    segs = []
    configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        color, depth, seg = env.render_camera(config)
        xyz = get_pointcloud(depth, config["intrinsics"])
        
        # xyz = get_pointcloud(depth, config["intrinsics"])
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        # transform pc from camera_base to robot_base
        points = transform_pointcloud(xyz, transform)
        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
        iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
        iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = color[valid]
        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        pcd.voxel_down_sample(reconstruction_config['voxel_size'])
        # # visualization
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        # the first pcd is the one for start fusion
        pcds.append(pcd)
        segs.append(seg)

    # _, fuse_pcd = process_pcds(pcds, reconstruction_config)
    fuse_pcd = process_pcds_test(pcds)
    # ply_global = furthest_point_sampling(fuse_pcd, n_samples=25000)
    # ply_global_for_eval = furthest_point_sampling(fuse_pcd, n_samples=18000)
    # pcd_global = o3d.geometry.PointCloud()
    # pcd_global.points = o3d.utility.Vector3dVector(ply_global)
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([pcd_global, frame])
    return fuse_pcd, segs[0]

def get_global_label_pc(env, target_id):
    pcds = []
    segs = []
    configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        # 渲染
        color, depth, seg = env.render_camera(config)

        # HxWx5: [x,y,z, l0, l1]，其中 l 为 one-hot（例如 [0,1] 代表目标，[0,0,1] 则是你三类时的第三列，按你实际为两类或三类来）
        xyz_label = global_label_points(depth, config["intrinsics"], seg, target_id)  # HxWx5

        # 拆分
        xyz_hw   = xyz_label[:, :, :3].astype(np.float64)   # HxWx3
        labels_hw = xyz_label[:, :, 3:].astype(np.float32)  # HxWx2
        H, W = depth.shape

        # 位姿变换（保持 HxWx3，再展平）
        position = np.array(config["position"]).reshape(3, 1)
        rotation = np.array(p.getMatrixFromQuaternion(config["rotation"])).reshape(3, 3)
        T = np.eye(4); T[:3, :3] = rotation; T[:3, 3] = position[:, 0]

        points_hw = transform_pointcloud(xyz_hw, T)         # HxWx3
        points    = points_hw.reshape(-1, 3)                # N x 3
        labels    = labels_hw.reshape(-1, labels_hw.shape[-1])  # N x C (C=2或3)
        colors    = color.reshape(-1, 3).astype(np.float64)      # N x 3

        # 工作空间过滤（labels/colors 同步）
        ix = (points[:, 0] >= env.bounds[0, 0]) & (points[:, 0] < env.bounds[0, 1])
        iy = (points[:, 1] >= env.bounds[1, 0]) & (points[:, 1] < env.bounds[1, 1])
        iz = (points[:, 2] >= env.bounds[2, 0]) & (points[:, 2] < env.bounds[2, 1])
        valid = ix & iy & iz

        points = points[valid]
        labels = labels[valid]
        colors = colors[valid]

        # 按 z 排序（明确用 [:,2]）
        order = np.argsort(points[:, 2])
        points = points[order]
        labels = labels[order]
        colors = colors[order]

        # Open3D 点云（仅几何/颜色）
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        # 体素降采样 + trace（拿原始索引）
        voxel = reconstruction_config['voxel_size']
        min_b = pcd.get_min_bound() - voxel
        max_b = pcd.get_max_bound() + voxel

        pcd_ds, _, traces = pcd.voxel_down_sample_and_trace(
            voxel_size=voxel,
            min_bound=min_b,
            max_bound=max_b,
            approximate_class=True
        )

        xyz_ds = np.asarray(pcd_ds.points).astype(np.float32)   # (M,3)
        # 继承标签：体素内第一个原始点的标签
        keep_idx = np.array([np.asarray(idx, dtype=np.int64)[0] for idx in traces], dtype=np.int64)
        # 若想用“距离体素输出点最近”的原始点标签：
        # keep_idx = np.array([ idxs[np.argmin(np.linalg.norm(points[np.asarray(idxs)] - xyz_ds[i], axis=1))]
        #                       for i, idxs in enumerate(traces) ], dtype=np.int64)
        # 若想用“多数投票”（one-hot 求和后 argmax）：
        # lab_ds = np.stack([labels[np.asarray(idxs)].sum(0) for idxs in traces], axis=0)
        # lab_ds = (lab_ds == lab_ds.max(axis=1, keepdims=True)).astype(np.float32)

        lab_ds = labels[keep_idx].astype(np.float32)           # (M,C)

        # 重新拼装 (M, 3+C)  -> 你的 C=2 时就是 (M,5)
        xyz_label_ds = np.concatenate([xyz_ds, lab_ds], axis=1).astype(np.float32)

        # # visualization
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        # the first pcd is the one for start fusion
        pcds.append(xyz_label_ds)
        segs.append(seg)

    # _, fuse_pcd = process_pcds(pcds, reconstruction_config)
    # fuse_pcd = process_pcds_test(pcds)
    points_state = np.vstack(pcds)

    ply_global = fps_xyz_label(points_state, n_samples=25000)
    # ply_global_for_eval = furthest_point_sampling(fuse_pcd, n_samples=18000)
    # pcd_global = o3d.geometry.PointCloud()
    # pcd_global.points = o3d.utility.Vector3dVector(ply_global)
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([pcd_global, frame])
    return ply_global, segs[0]

def get_pcd_for_all(env, obj_id):
    pcd_g = []
    pcd_o = []
    pcd_for_graspnet = []
    configs = [env.oracle_cams[0], env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        color, depth, seg = env.render_camera(config)
        xyz1, xyz2, xyz3 = get_all_pointcloud(depth, config["intrinsics"], seg, obj_id)
        # xyz = get_pointcloud(depth, config["intrinsics"])
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        all_xyz = [xyz1, xyz2, xyz3]
        i = 0
        for xyz in all_xyz:
            points = transform_pointcloud(xyz, transform)
            # Filter out 3D points that are outside of the predefined bounds.

            ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
            iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
            iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
            valid = ix & iy & iz
            points = points[valid]
            colors = color[valid]
            # Sort 3D points by z-value, which works with array assignment to simulate
            # z-buffering for rendering the heightmap image.
            iz = np.argsort(points[:, -1])
            points, colors = points[iz], colors[iz]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            pcd.voxel_down_sample(reconstruction_config['voxel_size'])
            # # visualization
            # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            # o3d.visualization.draw_geometries([pcd, frame])
            # the first pcd is the one for start fusion
            if(i == 0):
                pcd_g.append(pcd)
                i += 1
            elif(i == 1):
                pcd_for_graspnet.append(pcd)
                i += 1
            else:
                pcd_o.append(pcd)

    _, fuse_pcd1 = process_pcds(pcd_g, reconstruction_config)
    _, fuse_pcd2 = process_pcds(pcd_for_graspnet, reconstruction_config)
    _, fuse_pcd3 = process_pcds(pcd_o, reconstruction_config)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    o3d.visualization.draw_geometries([fuse_pcd1, frame])
    o3d.visualization.draw_geometries([fuse_pcd2, frame])
    o3d.visualization.draw_geometries([fuse_pcd3, frame])
    return fuse_pcd1, fuse_pcd2, fuse_pcd3

def get_global_pc_from_multi_view(env):
    """
    set two fixed cameras with 45-degree tilt angle to get dense pc.
    """
    pcds = []
    segs = []
    configs = [env.agent_cams[1], env.agent_cams[2]]
    
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        color, depth, seg = env.render_camera(config)
        xyz = get_pointcloud(depth, config["intrinsics"])
        # xyz = get_pointcloud(depth, config["intrinsics"])
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        # transform pc from camera_base to robot_base
        points = transform_pointcloud(xyz, transform)
        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
        iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
        iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
        valid = ix & iy & iz

        valid = np.isfinite(depth) & (depth > 0)            
        valid &= np.isfinite(points).all(axis=-1) 

        points = points[valid]
        colors = color[valid]
        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        # pcd.voxel_down_sample(reconstruction_config['voxel_size'])
        # # visualization
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        # the first pcd is the one for start fusion
        pcds.append(pcd)
        segs.append(seg)

    fuse_pcd = process_pcds_test(pcds)
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([pcd_global, frame])
    return fuse_pcd, pcd

def get_obj_pc_from_multi_view(env, obj_id):
    """
    set two fixed cameras with 45-degree tilt angle to get dense pc.
    """
    np.random.seed(1239)
    pcds = []
    configs = [env.agent_cams[1], env.agent_cams[2]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    for config in configs:
        color, depth, seg = env.render_camera(config)
        _, xyz = get_mask_pointcloud(depth, config["intrinsics"], seg, obj_id)
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        points = transform_pointcloud(xyz, transform)
        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= env.bounds[0, 0]) & (points[Ellipsis, 0] < env.bounds[0, 1])
        iy = (points[Ellipsis, 1] >= env.bounds[1, 0]) & (points[Ellipsis, 1] < env.bounds[1, 1])
        iz = (points[Ellipsis, 2] >= env.bounds[2, 0]) & (points[Ellipsis, 2] < env.bounds[2, 1])
        valid = ix & iy & iz

        # no filter
        valid = (seg == obj_id)
        valid &= np.isfinite(depth) & (depth > 0)
        valid &= np.isfinite(points).all(axis=-1)
        valid &= (np.linalg.norm(points, axis=-1) > 1e-9)

        points = points[valid]
        colors = color[valid]
        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.lexsort((points[:,1], points[:,0], points[:,2]))
        points, colors = points[iz], colors[iz]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        # pcd.voxel_down_sample(reconstruction_config['voxel_size'])
        # # visualization
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        # the first pcd is the one for start fusion
        pcds.append(pcd)

    # _, fuse_pcd = process_pcds(pcds, reconstruction_config)
    fuse_pcd = process_pcds_test(pcds)
    # visualization
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([fuse_pcd, frame])

    return fuse_pcd, pcd

def adjust_pose_z_axis_to_down(rot_matrix):

    # 保留原始 X 轴方向（可选）
    x_axis = rot_matrix[:, 0]  # shape (3,)

    # 目标 Z 轴为竖直向下
    z_axis = np.array([0, 0, -1])

    # 重新计算 Y，使得 XYZ 成为右手坐标系
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 重新计算正交的 X
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # 构造新旋转矩阵
    new_rot = np.stack([x_axis, y_axis, z_axis], axis=1)  # 每列是一个轴

    # 转为四元数
    new_quat = R.from_matrix(new_rot).as_quat()  # [x, y, z, w]
    return new_quat, new_rot

def furthest_point_sampling(points, colors=None, semantics=None, n_samples=4096):
    """
    points: [N, 3] tensor containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically &lt;&lt; N
    """
    # Convert points to PyTorch tensor if not already and move to GPU
    # print(colors)
    pcd_np = np.asarray(points.points)
    # pcd_np = points.cpu().numpy()
    points = torch.from_numpy(pcd_np).float().cuda()  # [N, 3]
    # points = points.to('cuda')
    if colors is not None:
        colors = torch.Tensor(colors).cuda()
    if semantics is not None:
        semantics = semantics.astype(np.int32)
        semantics = torch.Tensor(semantics).cuda()

    # Number of points
    num_points = points.size(0)  # N

    # Initialize an array for the sampled indices
    sample_inds = torch.zeros(n_samples, dtype=torch.long).cuda()  # [S]

    # Initialize distances to inf
    dists = torch.ones(num_points).cuda() * float("inf")  # [N]

    # Select the first point randomly
    selected = torch.randint(num_points, (1,), dtype=torch.long).cuda()  # [1]
    sample_inds[0] = selected

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        last_added = sample_inds[i - 1]  # Scalar
        dist_to_last_added_point = torch.sum(
            (points[last_added] - points) ** 2, dim=-1
        )  # [N]

        # If closer, update distances
        dists = torch.min(dist_to_last_added_point, dists)  # [N]

        # Pick the one that has the largest distance to its nearest neighbor in the sampled set
        selected = torch.argmax(dists)  # Scalar
        sample_inds[i] = selected

    if colors is not None and semantics is not None:
        return (
            points[sample_inds].cpu().numpy(),
            colors[sample_inds].cpu().numpy(),
            semantics[sample_inds].cpu().numpy(),
        )  # [S, 3]
    elif colors is not None:
        return points[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy()
    else:
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points[sample_inds].cpu().numpy())
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
        return points[sample_inds]

def fps_xyz_label(xyz_label: np.ndarray,
                  n_samples: int = 4096,
                  return_index: bool = False,
                  device: str = None,
                  start_idx: int = None,
                  seed: int = None):
    """
    Furthest Point Sampling on xyz and gather corresponding one-hot labels.

    Args:
        xyz_label: (N, 3+C) numpy array, first 3 are xyz, remaining C are labels (one-hot).
                   你的场景: C=2 -> (N,5)
        n_samples: 输出点数 S，自动截断到 N 以内
        return_index: 是否额外返回采样到的原始索引 (S,)
        device: 'cuda' 或 'cpu'；默认自动选择（DataLoader worker 内建议显式传 'cpu'）
        start_idx: 首个点的索引；为 None 时随机选择
        seed: 若需要确定性随机起点，给一个种子

    Returns:
        sampled: (S, 3+C) numpy array
        (可选) idx: (S,) numpy int64 indices
    """
    arr = np.asarray(xyz_label)
    assert arr.ndim == 2 and arr.shape[1] >= 4, "xyz_label 应为 (N, 3+C)"

    N, D = arr.shape
    C = D - 3
    S = min(n_samples, N)

    # 设备选择
    # if device is None:
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # DataLoader worker 内如未使用 spawn，建议强制 cpu
    device = 'cuda'

    xyz = torch.from_numpy(arr[:, :3]).to(device=device, dtype=torch.float32)   # [N,3]
    lab = torch.from_numpy(arr[:, 3:]).to(device=device, dtype=torch.float32)   # [N,C]

    # 采样索引与距离
    idx = torch.empty(S, dtype=torch.long, device=device)
    dists = torch.full((N,), float('inf'), device=device)

    # 首点
    if start_idx is None:
        if seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(int(seed))
            idx0 = torch.randint(N, (1,), generator=g, device=device)[0]
        else:
            idx0 = torch.randint(N, (1,), device=device)[0]
    else:
        idx0 = torch.tensor(start_idx, dtype=torch.long, device=device).clamp_(0, N-1)

    idx[0] = idx0

    # 迭代选点（O(N*S)）
    for i in range(1, S):
        last = idx[i-1]
        dist2 = torch.sum((xyz - xyz[last])**2, dim=-1)  # [N]
        dists = torch.minimum(dists, dist2)
        idx[i] = torch.argmax(dists)

    # 收集结果
    sampled_xyz = xyz[idx]          # (S,3)
    sampled_lab = lab[idx]          # (S,C)
    sampled = torch.cat([sampled_xyz, sampled_lab], dim=1).cpu().numpy()  # (S, 3+C)

    if return_index:
        return sampled, idx.cpu().numpy().astype(np.int64)
    else:
        return sampled

def furthest_point_sampling_nocuda(points, colors=None, semantics=None, n_samples=4096,start_idx=0):
    """
    points: [N, 3] tensor containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically &lt;&lt; N
    """

    # Number of points
    num_points = points.shape[0] # N

    # Initialize an array for the sampled indices
    sample_inds = torch.zeros(n_samples, dtype=torch.long, device=points.device) # [S]

    # Initialize distances to inf
    dists = torch.ones(num_points) * float("inf")  # [N]

    # Select the first point randomly
    # selected = torch.randint(num_points, (1,), dtype=torch.long)  # [1]
    selected = torch.tensor([start_idx], dtype=torch.long, device=points.device)
    sample_inds[0] = selected

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        last_added = sample_inds[i - 1]  # Scalar
        dist_to_last_added_point = torch.sum(
            (points[last_added] - points) ** 2, dim=-1
        )  # [N]

        # If closer, update distances
        dists = torch.min(dist_to_last_added_point, dists)  # [N]

        # Pick the one that has the largest distance to its nearest neighbor in the sampled set
        selected = torch.argmax(dists)  # Scalar
        sample_inds[i] = selected

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points[sample_inds].cpu().numpy())
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        # o3d.visualization.draw_geometries([pcd, frame])
    return points[sample_inds], sample_inds



# def fps_p3d(points: torch.Tensor,
#                   n_samples: int,
#                   start_idx: int = 0,
#                   return_index: bool = True):
#     """
#     points: [N, C>=3] torch.Tensor
#     返回: out_points [n_samples, C]（始终固定长度）
#     """
#     from pytorch3d.ops import sample_farthest_points

#     assert isinstance(points, torch.Tensor), "points 必须是 torch.Tensor"
#     assert points.ndim == 2 and points.shape[1] >= 3, f"points 需为 [N,C>=3]，但得到 {points.shape}"

#     device = points.device
#     N = int(points.shape[0])
#     S = int(n_samples)

#     if N == 0:
#         raise ValueError("points 为空，无法 FPS")

#     K = min(S, N)

#     xyz = points[:, :3].to(dtype=torch.float32).contiguous().unsqueeze(0)  # [1,N,3]

#     # 固定起点：交换到第 0 个，再 random_start_point=False
#     if start_idx is not None:
#         si = int(start_idx)
#         if not (0 <= si < N):
#             raise ValueError(f"start_idx 超界: {start_idx} (总点数 {N})")

#         perm = torch.arange(N, device=device)
#         if si != 0:
#             perm[0], perm[si] = perm[si], perm[0]
#         xyz_perm = xyz[:, perm, :]
#         _, idx_perm = sample_farthest_points(xyz_perm, K=K, random_start_point=False)
#         sel = perm[idx_perm[0]]  # [K]
#     else:
#         _, idx = sample_farthest_points(xyz, K=K, random_start_point=False)
#         sel = idx[0]  # [K]

#     if K < S:
#         pad = sel.new_full((S - K,), sel[0].item())
#         sel = torch.cat([sel, pad], dim=0)

#     out = points[sel]  # [S, C]
#     return (out, sel) if return_index else out

def fps_p3d(
    points: torch.Tensor,
    n_samples: int,
    start_idx: int = None,
    return_index: bool = True,
):
    """
    points: [N, C>=3] torch.Tensor
    返回: out_points [S, C]（固定长度）
    """
    from pytorch3d.ops import sample_farthest_points

    assert isinstance(points, torch.Tensor), "points 必须是 torch.Tensor"
    assert points.ndim == 2 and points.shape[1] >= 3, f"points 需为 [N,C>=3]，但得到 {points.shape}"

    device = points.device
    N = int(points.shape[0])
    S = int(n_samples)

    if N == 0:
        raise ValueError("points 为空，无法 FPS")

    # 1) 当 N <= S 时，没必要做 FPS：直接保留全部点，再 pad
    if N <= S:
        sel = torch.arange(N, device=device, dtype=torch.long)
        if N < S:
            pad = sel[:1].expand(S - N)   # 避免 item() 同步
            sel = torch.cat([sel, pad], dim=0)
        out = points[sel]
        return (out, sel) if return_index else out

    # 2) 只在需要时转换 dtype / contiguous
    xyz = points[:, :3]
    if xyz.dtype != torch.float32:
        xyz = xyz.float()
    if not xyz.is_contiguous():
        xyz = xyz.contiguous()
    xyz = xyz.unsqueeze(0)  # [1, N, 3]

    # 3) 默认不重排；只有确实指定非 0 起点时才重排
    if start_idx is None or int(start_idx) == 0:
        _, idx = sample_farthest_points(xyz, K=S, random_start_point=False)
        sel = idx[0]
    else:
        si = int(start_idx)
        if not (0 <= si < N):
            raise ValueError(f"start_idx 超界: {start_idx} (总点数 {N})")

        perm = torch.arange(N, device=device)
        perm0 = perm[0].clone()
        permi = perm[si].clone()
        perm[0] = permi
        perm[si] = perm0

        xyz_perm = xyz[:, perm, :]
        _, idx_perm = sample_farthest_points(xyz_perm, K=S, random_start_point=False)
        sel = perm[idx_perm[0]]

    out = points[sel]
    return (out, sel) if return_index else out

def pad_pointcloud_list(points_list, pad_value: float = 0.0):
    """
    points_list: list of [Ni, C] torch.Tensor, Ni 可变, C 相同
    返回:
        padded : [B, Nmax, C]
        lengths: [B]
    """
    assert isinstance(points_list, (list, tuple)) and len(points_list) > 0, "points_list 不能为空"

    first = points_list[0]
    assert isinstance(first, torch.Tensor) and first.ndim == 2, "每个元素都必须是 [N,C] 的 torch.Tensor"

    device = first.device
    dtype = first.dtype
    C = first.shape[1]
    B = len(points_list)

    lengths = torch.empty((B,), dtype=torch.long, device=device)
    max_n = 0

    for i, p in enumerate(points_list):
        assert isinstance(p, torch.Tensor), f"第 {i} 个元素不是 torch.Tensor"
        assert p.ndim == 2, f"第 {i} 个元素维度错误: {p.shape}"
        assert p.shape[1] == C, f"第 {i} 个元素通道数不一致: {p.shape[1]} vs {C}"
        assert p.device == device, f"第 {i} 个元素 device 不一致"
        lengths[i] = p.shape[0]
        if p.shape[0] > max_n:
            max_n = p.shape[0]

    if max_n <= 0:
        raise ValueError("所有点云都为空，无法打包")

    padded = torch.full((B, max_n, C), pad_value, dtype=dtype, device=device)
    for i, p in enumerate(points_list):
        n = p.shape[0]
        if n > 0:
            padded[i, :n] = p

    return padded, lengths

def fps_p3d_batch_padded(
    points_padded: torch.Tensor,
    lengths: torch.Tensor,
    n_samples: int,
    start_idx=None,
    return_index: bool = True,
):
    """
    points_padded: [B, Nmax, C>=3]
    lengths      : [B]
    返回:
        out : [B, S, C]
        idx : [B, S]
    """
    from pytorch3d.ops import sample_farthest_points

    assert isinstance(points_padded, torch.Tensor), "points_padded 必须是 torch.Tensor"
    assert points_padded.ndim == 3 and points_padded.shape[2] >= 3, \
        f"points_padded 需为 [B,N,C>=3]，但得到 {points_padded.shape}"
    assert isinstance(lengths, torch.Tensor), "lengths 必须是 torch.Tensor"
    assert lengths.ndim == 1 and lengths.shape[0] == points_padded.shape[0], \
        f"lengths 需为 [B]，但得到 {lengths.shape}"

    device = points_padded.device
    B, Nmax, C = points_padded.shape
    S = int(n_samples)

    lengths = lengths.to(device=device, dtype=torch.long)
    if torch.any(lengths <= 0):
        raise ValueError("lengths 中存在 <= 0 的项，无法 FPS")

    # start_idx 当前先支持 None / 0 / 全 0；
    # 若你后面确实要每个 batch 用不同起点，再单独扩展。
    if start_idx is not None:
        if isinstance(start_idx, int):
            if start_idx != 0:
                raise NotImplementedError("batched FPS 暂不支持统一非 0 start_idx")
        elif isinstance(start_idx, torch.Tensor):
            if torch.any(start_idx.to(device=device, dtype=torch.long) != 0):
                raise NotImplementedError("batched FPS 暂不支持逐 batch 非 0 start_idx")
        else:
            raise TypeError("start_idx 只能是 None / int / torch.Tensor")

    xyz = points_padded[:, :, :3]
    if xyz.dtype != torch.float32:
        xyz = xyz.float()
    if not xyz.is_contiguous():
        xyz = xyz.contiguous()

    idx_out = torch.empty((B, S), dtype=torch.long, device=device)

    # 1) 小点云：完全保持你原来的单点逻辑
    small_mask = lengths <= S
    if torch.any(small_mask):
        small_ids = torch.nonzero(small_mask, as_tuple=False).flatten()
        for b in small_ids.tolist():
            n = int(lengths[b].item())
            sel = torch.arange(n, device=device, dtype=torch.long)
            if n < S:
                pad = sel[:1].expand(S - n)
                sel = torch.cat([sel, pad], dim=0)
            idx_out[b] = sel

    # 2) 大点云：真正 batched FPS
    large_mask = lengths > S
    if torch.any(large_mask):
        large_ids = torch.nonzero(large_mask, as_tuple=False).flatten()
        xyz_large = xyz[large_ids]              # [Bl, Nmax, 3]
        lengths_large = lengths[large_ids]      # [Bl]

        _, idx_large = sample_farthest_points(
            xyz_large,
            lengths=lengths_large,
            K=S,
            random_start_point=False,
        )  # [Bl, S]

        idx_out[large_ids] = idx_large

    batch_ids = torch.arange(B, device=device)[:, None]
    out = points_padded[batch_ids, idx_out]     # [B, S, C]

    return (out, idx_out) if return_index else out

def fps_p3d_batch_from_list(
    points_list,
    n_samples: int,
    start_idx=None,
    return_index: bool = True,
    pad_value: float = 0.0,
):
    """
    points_list: list of [Ni, C]
    返回:
        out     : [B, S, C]
        idx     : [B, S]
        lengths : [B]
    """
    padded, lengths = pad_pointcloud_list(points_list, pad_value=pad_value)
    ret = fps_p3d_batch_padded(
        padded,
        lengths,
        n_samples=n_samples,
        start_idx=start_idx,
        return_index=return_index,
    )
    if return_index:
        out, idx = ret
        return out, idx, lengths
    else:
        out = ret
        return out, lengths
    
def voxel_downsample_keep_one_torch(points: torch.Tensor, voxel_size: float):
    """
    确定性体素降采样：每个 voxel 保留一个真实点（按原始顺序的第一个点）
    points: [N, C], 前3维是 xyz
    return:
        ds_points: [M, C]
        keep_idx:  [M] 在原 points 中的索引
    """
    assert points.ndim == 2 and points.shape[1] >= 3
    assert voxel_size > 0

    device = points.device
    xyz = points[:, :3]
    if xyz.dtype != torch.float32:
        xyz = xyz.float()

    # 体素坐标
    voxel = torch.floor(xyz / voxel_size).to(torch.int64)   # [N, 3]

    # 平移到非负，便于做唯一编码
    voxel_min = voxel.min(dim=0).values
    voxel = voxel - voxel_min

    # 计算每一维跨度，做无碰撞编码
    span = voxel.max(dim=0).values + 1  # [3]
    sx, sy, sz = span.tolist()

    # 对你当前这种局部 crop 点云规模，这样编码一般是安全的
    voxel_id = voxel[:, 0] * (sy * sz) + voxel[:, 1] * sz + voxel[:, 2]   # [N]

    # 关键点：为了确定性，按 (voxel_id, 原始索引) 排序
    # 这样同一 voxel 内一定保留原始顺序最靠前的那个点
    idx = torch.arange(points.shape[0], device=device, dtype=torch.int64)
    key = voxel_id * (points.shape[0] + 1) + idx
    order = torch.argsort(key)

    voxel_sorted = voxel_id[order]
    keep_mask = torch.ones_like(voxel_sorted, dtype=torch.bool)
    keep_mask[1:] = voxel_sorted[1:] != voxel_sorted[:-1]

    keep_idx = order[keep_mask]

    # 恢复到原始顺序，保持输出稳定
    keep_idx = torch.sort(keep_idx).values
    ds_points = points[keep_idx]
    return ds_points, keep_idx


def pre_voxel_downsample_for_fps(
    points: torch.Tensor,
    cap: int = 2200,
    trigger: int = 2400,
    voxel_size0: float = 0.0025,
    growth: float = 1.35,
    max_iter: int = 6,
):
    """
    只对超大点集做保守体素预采样，保持可复现。
    - N <= trigger: 原样返回
    - N > trigger : 逐步增大 voxel_size，直到点数降到 cap 附近
    """
    assert points.ndim == 2 and points.shape[1] >= 3
    N = points.shape[0]
    device = points.device

    if N <= trigger:
        idx = torch.arange(N, device=device, dtype=torch.long)
        return points, idx, 0.0

    xyz = points[:, :3]
    if xyz.dtype != torch.float32:
        xyz = xyz.float()

    # 用 bbox 对角线给一个保守初值，避免不同 crop 尺度差异太大
    bbox_min = xyz.min(dim=0).values
    bbox_max = xyz.max(dim=0).values
    diag = torch.linalg.norm(bbox_max - bbox_min).item()

    voxel_size = max(voxel_size0, diag / 128.0)

    best_points = points
    best_idx = torch.arange(N, device=device, dtype=torch.long)

    for _ in range(max_iter):
        ds_points, ds_idx = voxel_downsample_keep_one_torch(points, voxel_size)
        best_points, best_idx = ds_points, ds_idx

        if ds_points.shape[0] <= cap:
            break

        voxel_size *= growth

    # 如果体素后仍然太多，再做一个确定性的均匀索引抽样兜底
    # 这里只是极端情况兜底，正常一般到不了这一步
    if best_points.shape[0] > cap:
        sel = torch.linspace(
            0, best_points.shape[0] - 1, steps=cap, device=device
        ).long()
        best_idx = best_idx[sel]
        best_points = points[best_idx]

    return best_points, best_idx, voxel_size

def furthest_point_sampling_v2(points, colors=None, semantics=None, n_samples=4096):
    """
    points: [N, 3] tensor containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    # Convert points to PyTorch tensor if not already and move to GPU
    # points = torch.from_numpy(points).float().to(device='cuda')  # Automatically move to GPU if needed
    
    if colors is not None:
        colors = torch.Tensor(colors).to(device='cuda')  # Move colors to GPU
    if semantics is not None:
        semantics = torch.Tensor(semantics.astype(np.int32)).to(device='cuda')  # Move semantics to GPU

    # Number of points
    num_points = points.shape[0]  # N

    # Initialize an array for the sampled indices
    sample_inds = torch.zeros(n_samples, dtype=torch.long, device='cuda')  # [S]

    # Initialize distances to inf
    dists = torch.ones(num_points, device='cuda') * float("inf")  # [N]

    # Select the first point randomly
    selected = torch.randint(num_points, (1,), dtype=torch.long, device='cuda')  # [1]
    sample_inds[0] = selected

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        last_added = sample_inds[i - 1]  # Scalar
        dist_to_last_added_point = torch.sum((points[last_added] - points) ** 2, dim=-1)  # [N]

        # If closer, update distances
        dists = torch.min(dist_to_last_added_point, dists)  # [N]

        # Pick the one that has the largest distance to its nearest neighbor in the sampled set
        selected = torch.argmax(dists)  # Scalar
        sample_inds[i] = selected

    # Return the sampled points and corresponding attributes
    if colors is not None and semantics is not None:
        return (
            points[sample_inds].cpu().numpy(),
            colors[sample_inds].cpu().numpy(),
            semantics[sample_inds].cpu().numpy(),
        )  # [S, 3]
    elif colors is not None:
        return points[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy()
    else:
        return points[sample_inds].detach().cpu().numpy(), sample_inds

def write_ply(points, filename):
    """
    save 3D-points and colors into ply file.
    points: [N, 3] (X, Y, Z)
    filename: output filename
    """
    # combine vertices and colors
    vertices = np.array(
        [tuple(point) for point in points],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )

    el = PlyElement.describe(vertices, "vertex")

    # save PLY file
    PlyData([el], text=True).write(filename)

def write_npy(points, filename):
    """
    points: np.asarry
    filename: output filename
    """
    np.save(filename, points)
    # print(f"Saved: {filename}") 

def is_in_workplace(env,obj_num):
    is_in_workplace = True

    pos, _, _ = env.obj_info(obj_num)
    if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
        or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
        is_in_workplace = False
        print(f"\033[031m Target objects {obj_num} are not in the scene!\033[0m")
  
    return is_in_workplace

def grasp_pcd():

    finger1 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.015, depth=0.05)
    finger1.translate([-0.011, -0.0575 , -0.05])

    finger2 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.015, depth=0.05)
    finger2.translate([-0.011, 0.0425 , -0.05])

    finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.001)
    finger3.translate([-0.011, -0.0575 , -0.05])
    # finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.015)
    # finger3.translate([-0.011, -0.0575 , -0.065])

    gripper_mesh = finger1 + finger2 + finger3

    gripper_pcd = gripper_mesh.sample_points_poisson_disk(200) 

    gripper_points = torch.from_numpy(np.asarray(gripper_pcd.points)).float()

    return gripper_points, gripper_pcd

def hight_grasp_pcd():

    finger1 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.024, depth=0.0475)
    finger1.translate([-0.01, -0.0925 , -0.0475])

    finger2 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.024, depth=0.0475)
    finger2.translate([-0.01, 0.0685 , -0.0475])

    finger3 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.185, depth=0.05)
    # finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.001)
    # finger3.translate([-0.011, -0.0575 , -0.05])
    finger3.translate([-0.01, -0.0925 , -0.0975])

    gripper_mesh = finger1 + finger2 + finger3
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points[sample_inds].cpu().numpy())
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([gripper_mesh, frame])
    gripper_pcd = gripper_mesh.sample_points_uniformly(600) 
    # gripper_pcd = gripper_mesh.sample_points_poisson_disk(600) 

    gripper_points = torch.from_numpy(np.asarray(gripper_pcd.points)).float()

    # gripper_pcd.points = o3d.utility.Vector3dVector(gripper_points)

    return gripper_points

def push_gripper_pcd(z, n_target, oversample=2000, seed=55926):
    """
    gripper:[1,0,0]
    """
    finger1 = o3d.geometry.TriangleMesh.create_box(width=0.044, height=0.03, depth=0.05)
    finger1.translate([0.5 - 0.022, 0 - 0.015, z])
    # finger2 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.015, depth=0.05)
    # finger2.translate([0.5, -0.015, z])
    gripper_mesh = finger1 
    pts_dense = _area_weighted_sample_on_mesh(gripper_mesh, n_samples=oversample, seed=seed)  # [M,3] CPU

    # 2) 确定版 FPS 下采样
    pts_final = furthest_point_sampling_det(pts_dense, n_samples=n_target, start_idx=0)  # [n_target,3]

    # 3) Open3D点云（如需可视化）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_final.detach().cpu().numpy())

    return pts_final.float(), pcd

def TransformPCD2EndLink(point_cloud_base, pose):
    """
        将基坐标系下的点云变换到夹爪坐标系下。

        Parameters:
        - point_cloud_base: (N, 3) numpy array，基坐标系下的点云
        - T_base_to_gripper: (4, 4) numpy array，base -> gripper 的变换矩阵

        Returns:
        - point_cloud_gripper: (N, 3) numpy array，夹爪坐标系下的点云
    """
    # 构造齐次坐标 (N, 4)
    assert point_cloud_base.shape[1] == 3
    assert pose.shape[0] == 7

    device = point_cloud_base.device
    dtype = point_cloud_base.dtype
    # pose = torch.from_numpy(pose).float()
    # 提取旋转和平移
    position = pose[:3]  # [3]
    quat = pose[3:]      # [4]

    # 将四元数转换为旋转矩阵 (使用 torch 实现)
    qx, qy, qz, qw = quat
    R_mat = torch.tensor([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ], dtype=dtype, device=device)  # [3, 3]

    # 构造变换矩阵 T_base_to_gripper
    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = R_mat
    T[:3, 3] = position

    # 计算逆变换 T_gripper_to_base
    T_inv = torch.linalg.inv(T)  # [4, 4]

    # 齐次点云
    N = point_cloud_base.shape[0]
    ones = torch.ones((N, 1), dtype=dtype, device=device)
    points_homo = torch.cat([point_cloud_base, ones], dim=1)  # [N, 4]

    # 变换到 gripper 坐标系
    points_transformed = (T_inv @ points_homo.T).T  # [N, 4]
    return points_transformed[:, :3]

def Transform_Push2Fixed_point(global_pc, obj_pc, fixed_point, push_action):
    """
    All push actions must be normalized to a fixed reference point. 
    This ensures consistent left-to-right movement by the robot, which simplifies the learning process.
    """
    # push_pose = np.eye(4)
    # push_pose[:3,3] = push_action[:3]
    # push_pose[:3,:3] = R.from_quat(push_action[3:]).as_matrix()
    push_pose = torch.from_numpy(push_pose).float()
    fixed_pose = torch.eye(4)
    z = push_action[2]
    z = z.unsqueeze(-1)
    fixed_pose[:3,3] = torch.cat([fixed_point, z],dim=-1)
    fixed_pose[:3,:3] = torch.tensor([[0,-1,0],
                                      [-1,0,0],
                                      [0,0,-1]],dtype=float)
    T_2fixed =fixed_pose @ torch.linalg.inv(push_pose)
    obj_pc = torch.cat([obj_pc, torch.ones(obj_pc.shape[0], 1)],dim=1)
    global_pc = torch.cat([global_pc, torch.ones(global_pc.shape[0], 1)],dim=1)
    global_pc = (T_2fixed @ global_pc.T).T # NX4   
    obj_pc = (T_2fixed @ obj_pc.T).T # NX4 

    return global_pc[:,:3], obj_pc[:, :3]

def fuse_state(global_points,gripper_pcd):
    """
        1. grasp pose represented by point cloud;
        2. contact area when gripper close;
        3. goal object;
    """
    finger1 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.01, depth=0.05)
    finger1.translate([-0.011, -0.0425 , -0.05])

    finger2 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.01, depth=0.05)
    finger2.translate([-0.011, 0.0425 , -0.05])

    finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.085, depth=0.001)
    finger3.translate([-0.011, -0.0425 , -0.05])

    gripper_mesh = finger1 + finger2 + finger3

    grasp_pcd = gripper_mesh.sample_points_poisson_disk(200) 
    grasp_obb = grasp_pcd.get_oriented_bounding_box()

    global_pcd = o3d.geometry.PointCloud()
    global_pcd.points = o3d.utility.Vector3dVector(global_points)

    pcd_in_gripper = global_pcd.crop(grasp_obb)

    points_in_gripper = np.asarray(pcd_in_gripper.points)

    fuse_points = np.vstack([points_in_gripper, gripper_pcd])

    return fuse_points

def fuse_state_torch(global_points: torch.Tensor,
                     obj_points: torch.Tensor,
                     gripper_pcd: torch.Tensor,
                     threshold: float = 0.0065) -> torch.Tensor:
    """
    将 gripper 区域内与 obj 接触不充分的 global 点提取出来，与 gripper + obj 融合作为最终状态输入。
    
    Args:
        global_points: [N1, 3] 点云 (torch.Tensor)
        obj_points: [N2, 3] 点云 (torch.Tensor)
        gripper_pcd: [N3, 3] 点云 (torch.Tensor)
        threshold: float，距离阈值（用于判定 global 点是否接触 obj）

    Returns:
        fuse_points: [N_fuse, 3] torch.Tensor
    """
    device = global_points.device
    dtype = global_points.dtype

    # === 定义 Gripper 包围盒（仿 Open3D 结构） ===
    # 简化处理：统一用 AABB [x_min, x_max], [y_min, y_max], [z_min, z_max]
    # 模拟三个 finger 包围范围
    gripper_point = gripper_pcd.detach().cpu().numpy()
    gripper_pc = o3d.geometry.PointCloud()
    gripper_pc.points = o3d.utility.Vector3dVector(gripper_point)
    gripper_obb = gripper_pc.get_oriented_bounding_box()
    global_points = global_points.detach().cpu().numpy()
    global_pcd = o3d.geometry.PointCloud()
    global_pcd.points = o3d.utility.Vector3dVector(global_points)
    points_in_gripper = global_pcd.crop(gripper_obb)
    points_in_gripper = torch.from_numpy(np.asarray(points_in_gripper.points)).float()

    # x_range = [-0.02, 0.02]
    # y_range = [-0.05, 0.05]
    # z_range = [-0.085, 0.005]  # 向上留一点 margin

    # mask_x = (global_points[:, 0] >= x_range[0]) & (global_points[:, 0] <= x_range[1])
    # mask_y = (global_points[:, 1] >= y_range[0]) & (global_points[:, 1] <= y_range[1])
    # mask_z = (global_points[:, 2] >= z_range[0]) & (global_points[:, 2] <= z_range[1])

    # mask = mask_x & mask_y & mask_z
    # points_in_gripper = global_points[mask]  # [M, 3]

    # === 判断 points_in_gripper 是否接触到 obj_points（欧氏距离 < threshold）===
    if points_in_gripper.shape[0] == 0:
        contactless_points = torch.empty((0, 3), device=device, dtype=dtype)
    else:
        dist = torch.cdist(points_in_gripper.unsqueeze(0), obj_points.unsqueeze(0)).squeeze(0)  # [M, N2]
        min_dist, _ = torch.min(dist, dim=1)  # [M]
        mask_contactless = min_dist > threshold
        contactless_points = points_in_gripper[mask_contactless]  # [K, 3]
        

    # === 融合为最终状态 ===
    # fuse_points = torch.cat([contactless_points, gripper_pcd, obj_points], dim=0)  # [N_fuse, 3]
    fuse_points = torch.cat([contactless_points, obj_points], dim=0)  # [N_fuse, 3]
    return fuse_points

def extend_obb_single_dir_along_global_z(pcd: o3d.geometry.PointCloud, factor: float = 10.0):
    """
    将 pcd 的 OBB 沿全局 Z 轴反方向单侧延长为原来的 factor 倍。
    仅改变对应那一条边长度，保持 OBB 的朝向与另外两条边长度不变。
    """
    assert factor >= 1.0, "factor 应 >= 1.0（延长）"

    # 1) 原始 OBB
    obb = pcd.get_oriented_bounding_box()

    R = obb.R.copy()                # 3x3，列为 OBB 局部轴（单位向量）
    extent = obb.extent.copy()      # [ex, ey, ez]，分别对应 R 的三列
    center = obb.center.copy()

    # 2) 找到与全局 Z 轴最接近的 OBB 局部轴
    z = np.array([0.0, 0.0, 1.0])   # 全局 Z 轴
    dots = np.array([np.dot(R[:, i], z) for i in range(3)])     # 与 Z 的对齐度
    k = int(np.argmax(np.abs(dots)))                             # 最接近 Z 的那一列索引（0/1/2）

    old_len = extent[k]
    new_len = factor * old_len
    delta = new_len - old_len       # 需要增加的长度

    # 3) 单方向：沿全局 Z 的反方向（-Z）
    #    需要判断 R[:,k] 相对 Z 的朝向，选择与 -Z 同向的局部方向。
    #    若 R[:,k]·Z > 0，则 -Z 与 -R[:,k] 同向；否则与 +R[:,k] 同向。
    if dots[k] > 0:
        dir_vec = -R[:, k]
    else:
        dir_vec =  R[:, k]

    # 4) 调整中心与长度（只把“底部”那一面往 dir_vec 拉开，顶部保持不动）
    center = center + 0.5 * delta * dir_vec
    extent[k] = new_len

    # 5) 生成新的 OBB
    new_obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
    new_obb.color = (1, 0, 0)  # 红色：延长后的 OBB
    obb.color = (0, 0, 1)      # 蓝色：原 OBB

    return obb, new_obb

def fuse_state_torch_v2(global_points: torch.Tensor,
                     gripper_pcd: torch.Tensor,
                     threshold: float = 0.0065,
                     ):

    device = global_points.device
    dtype = global_points.dtype

    gripper_point = gripper_pcd.detach().cpu().numpy()
    gripper_pc = o3d.geometry.PointCloud()
    gripper_pc.points = o3d.utility.Vector3dVector(gripper_point)

    # gripper_obb = gripper_pc.get_oriented_bounding_box()
    _,gripper_obb = extend_obb_single_dir_along_global_z(gripper_pc)

    global_points = global_points.detach().cpu().numpy()
    global_pcd = o3d.geometry.PointCloud()
    global_pcd.points = o3d.utility.Vector3dVector(global_points)
    points_in_gripper = global_pcd.crop(gripper_obb)
    points_in_grippers = torch.from_numpy(np.asarray(points_in_gripper.points)).to(device=device, dtype=dtype)

    # world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    # o3d.visualization.draw_geometries([world, gripper_pc, global_pcd, gripper_obb])
    # o3d.visualization.draw_geometries([world, gripper_pc, points_in_gripper])
    # o3d.visualization.draw_geometries([world, obj_pcd,points_without_obj])
    # o3d.visualization.draw_geometries([world, gripper_pc, obj_pcd,points_without_obj])
    # o3d.visualization.draw_geometries([world, gripper_pc, points_in_gripper_without_obj,obj_pcd_within_gripper])

    # points_in_gripper = torch.from_numpy(np.asarray(points_in_gripper.points)).float()

    fuse_points = points_in_grippers

    return fuse_points

def fuse_state_torch_v3(global_points: torch.Tensor,
                        gripper_pcd: torch.Tensor,
                        threshold: float = 0.0065,
                        ):
    """
    global_points: (N, 3) torch.Tensor
    gripper_pcd:   (M, 3) torch.Tensor
    返回:
        fuse_points:   (K, 3) 裁剪后的点（仍在 global_points 坐标系）
        crop_indices:  (K,)   这些点在 global_points 中的索引
    """
    device = global_points.device
    dtype  = global_points.dtype

    # ---- 构造 gripper 点云 & OBB ----
    gripper_np = gripper_pcd.detach().cpu().numpy()
    gripper_pc = o3d.geometry.PointCloud()
    gripper_pc.points = o3d.utility.Vector3dVector(gripper_np)

    # 你的扩展 OBB 函数
    _, gripper_obb = extend_obb_single_dir_along_global_z(gripper_pc)

    # ---- 构造 global 点云 ----
    global_np = global_points.detach().cpu().numpy()
    global_pcd = o3d.geometry.PointCloud()
    global_pcd.points = o3d.utility.Vector3dVector(global_np)

    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # o3d.visualization.draw_geometries([gripper_pc, global_pcd, frame])
    # 关键：直接拿“在 OBB 内部的点索引”
    idx_in_obb = gripper_obb.get_point_indices_within_bounding_box(global_pcd.points)
    # idx_in_obb: Python list[int]，对应 global_np 的行索引

    if len(idx_in_obb) == 0:
        # 没有点落在 OBB 内，返回空
        crop_indices = torch.empty(0, dtype=torch.long, device=device)
        fuse_points  = global_points[crop_indices]   # (0, 3)
        return fuse_points, crop_indices

    crop_indices = torch.as_tensor(idx_in_obb, dtype=torch.long, device=device)  # (K,)
    fuse_points  = global_points[crop_indices]  # 直接从原始 Tensor 按索引取，避免再来回转换

    return fuse_points, crop_indices

def fuse_state(global_points: torch.Tensor,
                     gripper_pcd: torch.Tensor,) -> torch.Tensor:
    """
    将 gripper 区域内与 obj 接触不充分的 global 点提取出来，与 gripper + obj 融合作为最终状态输入。
    
    Args:
        global_points: [N1, 3] 点云 (torch.Tensor)
        gripper_pcd: [N3, 3] 点云 (torch.Tensor)

    Returns:
        fuse_points: [N_fuse, 3] torch.Tensor
    """
    device = global_points.device
    dtype = global_points.dtype

    # x_range = [-0.022, 0.022]
    # y_range = [-0.05, 0.05]
    # z_range = [-0.085, 0.005] 
    
    x_range = [-0.011, 0.011]
    y_range = [-0.0425, 0.0525]
    z_range = [-0.05, 0.0] # 向上留一点 margin

    mask_x = (global_points[:, 0] >= x_range[0]) & (global_points[:, 0] <= x_range[1])
    mask_y = (global_points[:, 1] >= y_range[0]) & (global_points[:, 1] <= y_range[1])
    mask_z = (global_points[:, 2] >= z_range[0]) & (global_points[:, 2] <= z_range[1])

    mask = mask_x & mask_y & mask_z
    points_in_gripper = global_points[mask]  # [M, 3]

    fuse_points = torch.cat([points_in_gripper, gripper_pcd], dim=0)  # [N_fuse, 3]
    return fuse_points

def natural_key(s):
    # 提取字符串中的整数用于排序，例如 "ply_global_10.ply" -> ['ply_global_', 10, '.ply']
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def uniform_point_count(points, target_n=345, jitter_std=0.00001):
    """
    points: np.ndarray of shape (N, 3)
    Returns: np.ndarray of shape (target_n, 3)
    """
    N = points.shape[0]
    
    if N > target_n:
        # 下采样：随机采样
        idx = np.random.choice(N, target_n, replace=False)
        return points[idx]
    
    elif N < target_n:
        # 上采样：重复采样 + 可选扰动
        idx = np.random.choice(N, target_n, replace=True)
        sampled = points[idx]
        
        # 可选：添加微小扰动，防止重复点完全重合
        noise = np.random.normal(0, jitter_std, size=sampled.shape)
        return sampled + noise
    
    else:
        return points
    
import torch

def uniform_point_count_torch(points: torch.Tensor, target_n: int = 240, jitter_std: float = 1e-6) -> torch.Tensor:
    """
    对点云进行上/下采样，使其统一为 target_n 个点，适用于 torch.Tensor 输入。

    Args:
        points: [N, 3] torch.Tensor，原始点云
        target_n: int，目标点数
        jitter_std: float，加性噪声标准差（仅用于上采样时防止重合）

    Returns:
        [target_n, 3] torch.Tensor，重采样后的点云
    """
    N = points.shape[0]
    device = points.device
    dtype = points.dtype

    if N > target_n:
        # 下采样
        idx = torch.randperm(N, device=device)[:target_n]
        return points[idx]

    elif N < target_n:
        # 上采样
        idx = torch.randint(0, N, (target_n,), device=device)
        sampled = points[idx]
        if jitter_std > 0:
            noise = torch.randn_like(sampled) * jitter_std
            sampled = sampled + noise
        return sampled

    else:
        return points

def pc_normalize(pc: torch.Tensor):
    """
    输入:
        pc: Tensor of shape [N, 3]，点云坐标，类型为 float32 或 float64
    返回:
        归一化后的点云，中心位于原点，最大半径为1
    """
    centroid = torch.mean(pc, dim=1)          # [3]
    pc = pc - centroid                         # 平移到原点
    m = torch.max(torch.sqrt(torch.sum(pc**2, dim=2)))  # 最远点距离
    pc = pc / m                                # 缩放归一化
    return pc

def pc_normalize_grasp(pc: torch.Tensor):
    """
    输入:
        pc: Tensor of shape [N, 3]，点云坐标，类型为 float32 或 float64
    返回:
        归一化后的点云，中心位于原点，最大半径为1
    """
    centroid = torch.mean(pc, dim=0)          # [3]
    pc = pc - centroid                         # 平移到原点
    m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))  # 最远点距离
    pc = pc / m                                # 缩放归一化
    return pc, centroid, m

def pc_normalize_for_obj(pc: torch.Tensor,
                        centroid: torch.Tensor,
                        m: torch.Tensor):
    """
    obj pc should used the same normalization way in global pc.
    """
    pc = pc - centroid
    pc = pc / m
    return pc

#-----------sample push action----------------------
def transform_points_to_camera(points_world, T_cam_base):
    num_points = points_world.shape[0]
    homo_points = np.hstack((points_world, np.ones((num_points, 1))))  # Nx4
    points_cam = (T_cam_base @ homo_points.T).T[:, :3]  # Nx3
    return points_cam

def project_points_to_image(points_cam, fx, fy, cx, cy):
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    Z[Z <= 0] = 1e-6  
    u = (X * fx / Z + cx).astype(int)
    v = (Y * fy / Z + cy).astype(int)
    return u, v

def dilate_masks(masks, kernel_size=3, iterations=1):
    H, W = masks.shape
    object_ids = np.unique(masks)
    object_ids = object_ids[object_ids != 0] # 排除背景

    dilated_mask = np.zeros_like(masks)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for obj_id in object_ids:
        binary_mask = (masks == obj_id).astype(np.uint8)
        dilated = cv2.dilate(binary_mask, kernel, iterations=iterations)
        dilated_mask[dilated > 0] = obj_id

    return dilated_mask

def segment_pointcloud_by_mask(points, masks):
    masks = dilate_masks(masks)
    intrinsics = np.array([[630000.0, 0, 320], [0, 630000.0, 240], [0, 0, 1]])  
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    position = np.array([0.5, 0, 1000.0]) 
    rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))
    rot_matrix = np.array(p.getMatrixFromQuaternion(rotation)).reshape(3, 3)

    T_base_to_cam = np.eye(4)
    T_base_to_cam[:3, :3] = rot_matrix
    T_base_to_cam[:3, 3] = position
    T_cam_base = np.linalg.inv(T_base_to_cam)

    # points = np.asarray(scene_pcd.points)
    H, W = masks.shape

    points_cam = transform_points_to_camera(points, T_cam_base)
    u, v = project_points_to_image(points_cam, fx, fy, cx, cy)

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u_valid = u[valid]
    v_valid = v[valid]
    points_valid = points[valid]
    labels = masks[v_valid, u_valid]

    object_ids = np.unique(labels)
    object_ids = object_ids[object_ids > 0]

    object_pcds = []
    object_masks = []
    for obj_id in object_ids[1:]:
        idx = np.where(labels == obj_id)[0]
        obj_points = points_valid[idx]

        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
        object_pcds.append(obj_pcd)

        mask = np.zeros((H,W),dtype=bool)
        mask[v_valid[idx],u_valid[idx]] = True
        object_masks.append(mask)

    return object_pcds,object_masks

def sample_surface_points(object_pcd,expand=0.016,step=0.03):
    points = np.asarray(object_pcd.points)
    aabb = object_pcd.get_axis_aligned_bounding_box()
    z_mean = (aabb.get_max_bound()[2] + aabb.get_min_bound()[2]) / 2
    xy = points[:, :2]
    resolution = 0.001
    xy_min = xy.min(axis=0)
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    pad = 10
    img_size = np.ceil((xy_max - xy_min) / resolution).astype(int) + 2*pad
    img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    indices = ((xy - xy_min) / resolution).astype(int) + pad
    img[indices[:, 1], indices[:, 0]] = 255 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    img_filled = binary_fill_holes(closed>0).astype(np.uint8) * 255
    dilated_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(img_filled, dilated_kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contour found from projected point cloud.")
    img_color = np.zeros((img_filled.shape[0],img_filled.shape[1],3),dtype=np.uint8)
    img_color[img_filled > 0] = [0,0,255]
    max_contour = max(contours, key=cv2.contourArea)
    # cv2.drawContours(img_color,max_contour,-1,(255,0,0),1)
    # cv2.imshow('Contours',img_color)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    contour_pts_image = max_contour[:, 0, :].astype(np.float32) 
    contour_pts_image -= pad
    contour_pts_world = contour_pts_image * resolution + xy_min
    poly = Polygon(contour_pts_world)
    offset_polygon = poly.buffer(expand)
    if isinstance(offset_polygon,MultiPolygon):
        offset_polygon = max(offset_polygon.geoms, key = lambda p: p.area)
    boundary_coords = np.array(offset_polygon.exterior.coords[:-1])  
    push_xy = interpolate_polygon_edges_with_step(boundary_coords,step)

    [0.276, 0.724], [-0.224, 0.224]
    sampled_points = []
    for uv_pt in push_xy:
        if 0.276 < uv_pt[0] < 0.724 and -0.224 < uv_pt[1] < 0.224:
            sampled_points.append([uv_pt[0], uv_pt[1], z_mean])

    return np.array(sampled_points)


def interpolate_polygon_edges_with_step(hull_pts, step=0.005):
    hull_pts = np.asarray(hull_pts, dtype=np.float32)
    n = len(hull_pts)
    if n < 2:
        return hull_pts.copy()

    # Step 1: 计算每条边的向量和长度
    edges = hull_pts[(np.arange(n) + 1) % n] - hull_pts # shape [n, 2]
    edge_lengths = np.linalg.norm(edges, axis=1)
    total_length = np.sum(edge_lengths)

    if total_length < 1e-6:
        return hull_pts[:1] # 全部重合时只返回一个点

    # Step 2: 计算累积长度
    cumulative_lengths = np.cumsum(edge_lengths)
    num_samples = max(int(np.floor(total_length / step)), 1)
    sample_distances = np.linspace(0, total_length, num_samples, endpoint=False)

    # Step 3: 插值采样点
    sampled_pts = []
    edge_idx = 0
    curr_edge_start = hull_pts[0]
    curr_edge_vec = edges[0]
    curr_edge_len = edge_lengths[0]
    curr_cum_len = 0.0

    for d in sample_distances:
    # 移动到对应边
        while d >= cumulative_lengths[edge_idx]:
            curr_cum_len = cumulative_lengths[edge_idx]
            edge_idx = (edge_idx + 1) % n
            curr_edge_start = hull_pts[edge_idx]
            curr_edge_vec = edges[edge_idx]
            curr_edge_len = edge_lengths[edge_idx]

        t = (d - curr_cum_len) / curr_edge_len # 当前边上的归一化位置
        pt = curr_edge_start + curr_edge_vec * t
        sampled_pts.append(pt)

    return np.array(sampled_pts)



def remove_points_near_other_cloud(pcd_A, pcd_B, radius):
    # A_points = np.asarray(pcd_A.points)
    # B_points = np.asarray(pcd_B.points)
    # B_tree = cKDTree(B_points[:, :2])
    # # 查询A点云中每个点在半径内的邻近点
    # neighbors = B_tree.query_ball_point(A_points[:, :2], r=radius, return_length=True)
    # # 没有邻近点的才保留
    # keep_mask = np.array(neighbors) == 0
    # filtered_pcd_A = o3d.geometry.PointCloud()
    # filtered_pcd_A.points = o3d.utility.Vector3dVector(A_points[keep_mask])

    # return filtered_pcd_A
    A_points = np.asarray(pcd_A.points)
    B_points = np.asarray(pcd_B.points)

    B_tree = cKDTree(B_points[:, :2])

    keep_mask = []
    for i in range(len(A_points)):
        a_xy = A_points[i, :2]
        a_z = A_points[i, 2]
        idxs = B_tree.query_ball_point(a_xy, r=radius)

        keep = True
        for j in idxs:
            if B_points[j, 2] > a_z:
                keep = False
                break
        keep_mask.append(keep)

    keep_mask = np.array(keep_mask)
    filtered_pcd_A = o3d.geometry.PointCloud()
    filtered_pcd_A.points = o3d.utility.Vector3dVector(A_points[keep_mask])

    return filtered_pcd_A

def compute_pose_dict(pcd_a,pcd_b):
    """
    计算点云A中每个点的姿态，返回每个点的pose字典
    
    Args:
    pcd_a: 点云A (open3d.geometry.PointCloud)
    pcd_b: 点云B (open3d.geometry.PointCloud)

    Returns:
    poses_dict: 包含每个点姿态的字典，键为点云A中的点索引，值为对应的4x4姿态矩阵
    """
    # 计算点云B的质心
    def compute_centroid(pcd):
        points = np.asarray(pcd.points)
        centroid = np.mean(points, axis=0)
        return centroid

    # 计算点云A中每个点的姿态
    def compute_pose(point, centroid_b):
        # 计算点到质心的向量
        direction = centroid_b - point
        # 投影到xy平面
        direction_xy = direction[:2]
        direction_xy /= np.linalg.norm(direction_xy)  # 归一化
        
        # x轴方向
        x_axis = np.array([direction_xy[0], direction_xy[1], 0])
        # y轴方向可以取竖直方向
        z_axis = np.array([0, 0, -1])  # 固定为竖直向下
        # z轴方向为x轴和y轴的叉积
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        # 计算旋转矩阵
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # 组合位姿
        pose = np.eye(4)
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = point

        return pose

    # 获取点云B的质心
    centroid_b = compute_centroid(pcd_b)

    # 保存点云A每个点的pose
    poses_dict = []

    # 遍历点云A中的每个点，计算并保存对应的姿态
    for idx, point in enumerate(np.asarray(pcd_a.points)):
        pose = compute_pose(point, centroid_b)
        poses_dict.append(pose)

    return poses_dict

def get_push_pose(object_pcd, pcd1):

    sample_points = sample_surface_points(object_pcd)
    if len(sample_points) == 0:
       poses_dict = []
    else:
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sample_points)
        filter_pcd = remove_points_near_other_cloud(sampled_pcd, pcd1, radius=0.015)
        # filter_pcd = remove_points_near_other_cloud(sampled_pcd, pcd1, radius=0.001)
        # poses_dict = compute_pose_dict(filter_pcd, object_pcd)
        poses_dict = compute_pose_dict(filter_pcd, object_pcd)
    return poses_dict


def sample_push_action(points,object_pcds):
    
    # points = np.asarray(scene_pcd.points)

    # 找到z坐标的最小值
    minZ = np.min(points[:, 2])

    # 筛选z坐标大于minZ+0.005的点
    indices = np.where(points[:, 2] >= minZ + 0.005)[0]
    new_points = points[indices]

    # 创建新的点云对象
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(new_points)
    # object_pcds,_ = segment_pointcloud_by_mask(points,masks)
    poses_dicts = []

    # 可视化聚类结果
    for i, object_pcd in enumerate(object_pcds):
        # o3d.visualization.draw_geometries([object_pcd])
        poses_dict = get_push_pose(object_pcd, pcd1)
        if len(poses_dict) == 0:
            continue
        poses_dicts.append(poses_dict)

    # secen_pcd = o3d.geometry.PointCloud()
    # secen_pcd.points = o3d.utility.Vector3dVector(points)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pcd1)
    # vis.add_geometry(secen_pcd)
    # for poses_dict in poses_dicts:  # 外层：每个聚类块
    #     for pose in poses_dict:     # 内层：该块内每个点的 pose
    #         coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.025)
    #         coordinate_frame.transform(pose)
    #         vis.add_geometry(coordinate_frame)

    # vis.run()
    # vis.destroy_window()
    
    return np.vstack(poses_dicts)
def save_grasp_action(global_pc, obj_pc, grasp_action, data_collect_id):

    ply_global_name = f"env_data_collection/adjust_data/global_pc/global_pc_{data_collect_id:05d}.ply"
    ply_obj_name = f"env_data_collection/adjust_data/obj_pc/ply_obj_{data_collect_id:05d}.ply"
    write_ply(global_pc, ply_global_name)
    write_ply(obj_pc, ply_obj_name)
    pose = np.hstack([grasp_action[:3, 3],R.from_matrix(grasp_action[:3, :3]).as_quat()])
    with open("env_data_collection/adjust_data/poses.txt", "a") as file:
                file.write(
                    f"{pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]}"
                    + "\n"
                )
    
    with open("env_data_collection/adjust_data/labels.txt", "a") as f:
        f.write(f"{int(0)}\n")
    
def transform_matrix2quat(push_actions):

    push_actions_sac = []
    for i in range(len(push_actions)):
        action = push_actions[i]
        position = action[:3, 3]
        rotation = action[:3, :3]
        r = R.from_matrix(rotation)
        quat = r.as_quat()
        t = np.hstack((position, quat))
        push_actions_sac.append(t)
    return np.vstack(push_actions_sac)

def transform_np2tensor(x):
    x = torch.from_numpy(x).float()
    x = x.unsqueeze(0) 
    x = x.transpose(1,2)
    return x



def _area_weighted_sample_on_mesh(mesh: o3d.geometry.TriangleMesh, n_samples: int, seed: int = 42) -> torch.Tensor:
    """
    在三角网格表面按面积“确定性随机”采样 n_samples 个点（受 seed 控制）。
    返回: [n_samples, 3] torch.float32 (CPU)
    """
    # 顶点与面
    V = np.asarray(mesh.vertices)        # [Nv, 3]
    F = np.asarray(mesh.triangles)       # [Nf, 3]

    # 面积
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    tri_areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    area_sum = tri_areas.sum()
    if area_sum <= 0:
        raise ValueError("Mesh has zero total area.")

    # 面选择的概率分布
    probs = tri_areas / area_sum

    # 受控随机源
    rng = np.random.RandomState(seed)

    # 采样面索引
    face_idx = rng.choice(len(F), size=n_samples, replace=True, p=probs)  # [n_samples]
    f0 = V[F[face_idx, 0]]
    f1 = V[F[face_idx, 1]]
    f2 = V[F[face_idx, 2]]

    # 三角形内的均匀采样（barycentric，sqrt trick）
    u = rng.rand(n_samples, 1)
    v = rng.rand(n_samples, 1)
    su = np.sqrt(u)
    w0 = 1.0 - su
    w1 = su * (1.0 - v)
    w2 = su * v

    pts = (w0 * f0) + (w1 * f1) + (w2 * f2)  # [n_samples, 3]
    return torch.from_numpy(pts.astype(np.float32))  # CPU tensor


def furthest_point_sampling_det(points: torch.Tensor, n_samples: int, start_idx: int = 0) -> torch.Tensor:
    """
    确定版 FPS（Euclidean）。输入 points:[N,3](CPU/GPU均可)，返回 [n_samples,3]，与设备一致。
    """
    if not torch.is_tensor(points):
        points = torch.tensor(points, dtype=torch.float32)
    device = points.device
    points = points.to(device=device, dtype=torch.float32)

    N = points.shape[0]
    n_samples = min(n_samples, N)

    sample_inds = torch.empty(n_samples, dtype=torch.long, device=device)
    dists = torch.full((N,), float("inf"), device=device)

    selected = torch.tensor([start_idx], dtype=torch.long, device=device)
    sample_inds[0] = selected

    for i in range(1, n_samples):
        last = sample_inds[i - 1]
        dist_to_last = torch.sum((points - points[last]) ** 2, dim=-1)
        dists = torch.minimum(dists, dist_to_last)
        selected = torch.argmax(dists)  # 确定的tie-break：返回第一个最大值
        sample_inds[i] = selected

    return points[sample_inds]


def grasp_pcd_bluenoise_like(n_target: int = 500, oversample: int = 5000, seed: int = 42, extend=0):
    """
    生成“更均匀、可复现”的抓手点云：
    1) 构网格 → 2) 面均匀过采样 oversample → 3) FPS 取 n_target → 4) 返回 torch.Tensor 与 Open3D 点云
    """
    # --- 构造你的夹爪网格（与你现有版本一致） ---
    finger1 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.015, depth=0.0475 + extend)
    finger1.translate([-0.01, -0.0835, -0.0475])
    finger2 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.015, depth=0.0475 + extend)
    finger2.translate([-0.01, 0.0685, -0.0475])
    finger3 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.167, depth=0.015)
    finger3.translate([-0.01, -0.0835, -0.0625])
    # finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.001)
    # finger3.translate([-0.011, -0.0575 , -0.05])
    gripper_mesh = finger1 + finger2 + finger3

    # 1) 受控过采样
    pts_dense = _area_weighted_sample_on_mesh(gripper_mesh, n_samples=oversample, seed=seed)  # [M,3] CPU

    # 2) 确定版 FPS 下采样
    pts_final = furthest_point_sampling_det(pts_dense, n_samples=n_target, start_idx=0)  # [n_target,3]

    # 3) Open3D点云（如需可视化）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_final.detach().cpu().numpy())

    return pts_final.float(), pcd

def gripper_point_width(n_target: int = 500, oversample: int = 5000, seed: int = 42, gripper_width=None):
    # --- 构造你的夹爪网格（与你现有版本一致） ---
    finger1 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.0143, depth=0.0475)
    finger1.translate([-0.01, -gripper_width/2 - 0.0143, -0.0475])
    finger2 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=0.0143, depth=0.0475)
    finger2.translate([-0.01, gripper_width/2, -0.0475])
    finger3 = o3d.geometry.TriangleMesh.create_box(width=0.02, height=gripper_width+0.0286, depth=0.0143)
    finger3.translate([-0.01, -gripper_width/2 - 0.0143, -0.0618])
    # finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.001)
    # finger3.translate([-0.011, -0.0575 , -0.05])
    gripper_mesh = finger1 + finger2 + finger3

    # 1) 受控过采样
    pts_dense = _area_weighted_sample_on_mesh(gripper_mesh, n_samples=oversample, seed=seed)  # [M,3] CPU

    # 2) 确定版 FPS 下采样
    pts_final = furthest_point_sampling_det(pts_dense, n_samples=n_target, start_idx=0)  # [n_target,3]

    # 3) Open3D点云（如需可视化）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_final.detach().cpu().numpy())

    return pts_final.float(), pcd

def extend_gripper_point(n_target: int = 500, oversample: int = 5000, seed: int = 42, gripper_width=None):
    extend = (gripper_width / 2) * math.sin(math.radians(24))
    finger1 = o3d.geometry.TriangleMesh.create_box(width=0.02 + extend * 2, height=0.015, depth=0.0475 + 0.01)
    finger1.translate([-0.01 - extend, -gripper_width/2 - 0.015, -0.0475  ])
    finger2 = o3d.geometry.TriangleMesh.create_box(width=0.02 + extend * 2, height=0.015, depth=0.0475 + 0.01)
    finger2.translate([-0.01 - extend, gripper_width/2, -0.0475 ])
    finger3 = o3d.geometry.TriangleMesh.create_box(width=0.02 + extend * 2, height=gripper_width + 0.03, depth=0.015)
    finger3.translate([-0.01 - extend, -gripper_width/2 - 0.015, -0.0625])
    # finger3 = o3d.geometry.TriangleMesh.create_box(width=0.022, height=0.115, depth=0.001)
    # finger3.translate([-0.011, -0.0575 , -0.05])
    gripper_mesh = finger1 + finger2 + finger3

    # 1) 受控过采样
    pts_dense = _area_weighted_sample_on_mesh(gripper_mesh, n_samples=oversample, seed=seed)  # [M,3] CPU

    # 2) 确定版 FPS 下采样
    pts_final = furthest_point_sampling_det(pts_dense, n_samples=n_target, start_idx=0)  # [n_target,3]

    # 3) Open3D点云（如需可视化）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_final.detach().cpu().numpy())
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    o3d.visualization.draw_geometries([pcd, frame])
    return pts_final.float(), pcd

def append_square_plane_voxel_cull(
    pcd: o3d.geometry.PointCloud,
    half_size: float = 0.2,           # 方形半边长（m），边长=2*half_size
    spacing: float = 0.001,           # 平面网格点间距（m）
    center_method: Literal["mean", "median"] = "mean",
    mode: Literal["xy", "3d"] = "xy", # 体素判定模式：xy 或 3d
    voxel_size: Union[float, Tuple[float, float, float]] = 0.002,  # 体素尺寸（m）
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),          # 体素网格原点（m）
    keep_attrs: bool = False,
    plane_color: Optional[Tuple[float, float, float]] = (0.6, 0.6, 0.6),
    set_plane_normals: bool = True,
) -> o3d.geometry.PointCloud:
    """
    生成位于 z=0 的正方形平面点云（规则网格），并用体素法剔除与输入点云“共用同一体素”的平面点，然后与输入点云合并返回。

    共用体素的定义:
      - mode="xy": floor((x - ox)/vx), floor((y - oy)/vy) 相同即视为共用体素
      - mode="3d": 上述再加 floor((z - oz)/vz) 也相同

    参数:
        pcd: 输入 Open3D 点云
        half_size: 方形半边长（m）
        spacing: 平面点网格间距（m）
        center_method: 确定平面中心 (cx,cy) 的方法：'mean' 或 'median'
        mode: 'xy' 或 '3d' 体素重叠判定模式
        voxel_size: 体素大小，可为标量或 (vx,vy,vz)
        origin: 体素网格原点 (ox,oy,oz)。体素化与原点有关，需固定以保证一致性
        keep_attrs: 是否为平面点补齐颜色/法向（仅当输入已具备对应属性时）
        plane_color: 平面点颜色 [0,1]
        set_plane_normals: 是否将平面点法向设置为 (0,0,1)

    返回:
        合并后的点云（输入点云 + 去重后的平面点）
    """
    if pcd.is_empty():
        raise ValueError("输入点云为空")

    pts = np.asarray(pcd.points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("点坐标应为 (N,3)")

    # 1) 平面中心
    if center_method == "mean":
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    elif center_method == "median":
        cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
    else:
        raise ValueError("center_method 仅支持 'mean' 或 'median'")
    cz = 0
    # cz = float(pts[:, 2].min())

    # 2) 生成正方形网格平面（z=0）
    xs = np.arange(-half_size, half_size + spacing, spacing) + cx
    ys = np.arange(-half_size, half_size + spacing, spacing) + cy
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    plane_pts = np.stack([xx, yy, np.full_like(xx, cz)], axis=-1).reshape(-1, 3)  # (M,3)

    # 3) 体素参数规范化
    if isinstance(voxel_size, (int, float)):
        vx = vy = vz = float(voxel_size)
    else:
        if len(voxel_size) != 3:
            raise ValueError("voxel_size 必须是标量或长度为3的元组")
        vx, vy, vz = map(float, voxel_size)

    ox, oy, oz = origin

    # 4) 计算输入点云的体素索引集合
    if mode == "xy":
        occ_ix = np.floor((pts[:, 0] - ox) / vx).astype(np.int64)
        occ_iy = np.floor((pts[:, 1] - oy) / vy).astype(np.int64)
        occ_indices = np.stack([occ_ix, occ_iy], axis=1)  # (N,2)
        plane_ix = np.floor((plane_pts[:, 0] - ox) / vx).astype(np.int64)
        plane_iy = np.floor((plane_pts[:, 1] - oy) / vy).astype(np.int64)
        plane_indices = np.stack([plane_ix, plane_iy], axis=1)  # (M,2)
        dim = 2
    elif mode == "3d":
        occ_ix = np.floor((pts[:, 0] - ox) / vx).astype(np.int64)
        occ_iy = np.floor((pts[:, 1] - oy) / vy).astype(np.int64)
        occ_iz = np.floor((pts[:, 2] - oz) / vz).astype(np.int64)
        occ_indices = np.stack([occ_ix, occ_iy, occ_iz], axis=1)  # (N,3)
        plane_ix = np.floor((plane_pts[:, 0] - ox) / vx).astype(np.int64)
        plane_iy = np.floor((plane_pts[:, 1] - oy) / vy).astype(np.int64)
        plane_iz = np.floor((plane_pts[:, 2] - oz) / vz).astype(np.int64)
        plane_indices = np.stack([plane_ix, plane_iy, plane_iz], axis=1)  # (M,3)
        dim = 3
    else:
        raise ValueError("mode 必须是 'xy' 或 '3d'")

    # 5) 利用“行视图”做高速集合判定：将 (k,dim) int64 转成无类型字节视图，再 in1d
    def _row_view(a: np.ndarray) -> np.ndarray:
        """将 int64 矩阵的每行视为一个字节记录，用于 in1d/集合判定"""
        if not a.flags['C_CONTIGUOUS']:
            a = np.ascontiguousarray(a)
        return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()

    occ_view = _row_view(occ_indices)
    occ_unique = np.unique(occ_view)  # 去重，减小集合规模
    plane_view = _row_view(plane_indices)
    overlap_mask = np.in1d(plane_view, occ_unique, assume_unique=True)  # True 表示共用体素，需要剔除
    keep_mask = ~overlap_mask

    plane_pts_kept = plane_pts[keep_mask]

    # 6) 合并并返回
    out_pts = np.concatenate([pts, plane_pts_kept], axis=0)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(out_pts.astype(np.float64))

    if keep_attrs:
        # 颜色
        if pcd.has_colors():
            in_cols = np.asarray(pcd.colors)
            if plane_color is None:
                plane_color = (0.5, 0.5, 0.5)
            plane_cols = np.tile(np.array(plane_color, dtype=np.float64)[None, :],
                                 (plane_pts_kept.shape[0], 1))
            out_cols = np.vstack([in_cols, plane_cols])
            out.colors = o3d.utility.Vector3dVector(out_cols)
        # 法向
        if pcd.has_normals():
            in_normals = np.asarray(pcd.normals)
            if set_plane_normals:
                plane_normals = np.tile(np.array([0.0, 0.0, 1.0]), (plane_pts_kept.shape[0], 1))
            else:
                plane_normals = np.zeros((plane_pts_kept.shape[0], 3))
            out_normals = np.vstack([in_normals, plane_normals])
            out.normals = o3d.utility.Vector3dVector(out_normals)

    return out


def furthest_point_sampling_onehot_nocuda(points, colors=None, semantics=None, n_samples=4096, start_idx=0):
    """
    points: [N, 6] = [x,y,z, onehot_3]，仅 xyz 参与FPS，返回 [S, 6]
    n_samples: samples you want in the sampled point cloud typically << N
    """
    # if colors is not None:
    #     colors = torch.Tensor(colors).cuda()
    if colors is not None:
        colors = torch.as_tensor(colors, dtype=torch.float32, device=points.device)
    if semantics is not None:
        semantics = torch.as_tensor(semantics.astype(np.int32), device=points.device)

    num_points = points.shape[0]  # N
    # n_samples = min(n_samples, num_points)

    sample_inds = torch.zeros(n_samples, dtype=torch.long, device=points.device)  # 保持在同device
    dists = torch.ones(num_points, device=points.device) * float("inf")

    # 仅用于距离计算的 xyz 视图 ------------- NEW
    xyz = points[:, :3].contiguous()  # [N,3]

    # 选择起点
    # selected = torch.randint(num_points, (1,), dtype=torch.long)  # [1]
    selected = torch.tensor(start_idx, dtype=torch.long, device=points.device)  # ---- 小修：标量
    sample_inds[0] = selected

    for i in range(1, n_samples):
        last_added = sample_inds[i - 1]
        # 用 xyz 计算距离 ---------------------- NEW
        dist_to_last_added_point = torch.sum((xyz[last_added] - xyz) ** 2, dim=-1)
        dists = torch.min(dist_to_last_added_point, dists)
        selected = torch.argmax(dists)
        sample_inds[i] = selected

    # 返回时用原始 points 按索引切片，带回 one-hot 与任何额外通道
    if colors is not None and semantics is not None:
        return (
            points[sample_inds].cpu().numpy(),     # [S,6] -------------- NEW: 原张量切片
            colors[sample_inds].cpu().numpy(),
            semantics[sample_inds].cpu().numpy(),
        )
    elif colors is not None:
        return points[sample_inds].cpu().numpy(), colors[sample_inds].cpu().numpy()
    else:
        return points[sample_inds]                 # [S,6]

def furthest_point_sampling_onehot_p3d(points, colors=None, semantics=None, n_samples=4096, start_idx=0):

    assert isinstance(points, torch.Tensor), "points 需为 torch.Tensor"
    device = points.device
    N = points.shape[0]
    K = min(int(n_samples), int(N))

    # xyz 视图 (B, P, 3)
    xyz = points[:, :3].to(dtype=torch.float32, device=device).contiguous().unsqueeze(0)  # [1, N, 3]

    # 若需要固定起点：把 start_idx 放到第一个位置，再跑 FPS
    if start_idx is not None:
        if not (0 <= start_idx < N):
            raise ValueError(f"start_idx 超界: {start_idx} (总点数 {N})")
        perm = torch.arange(N, device=device)
        # 交换 perm[0] 与 perm[start_idx]
        if start_idx != 0:
            perm0 = perm[0].clone()
            perm[0] = perm[start_idx]
            perm[start_idx] = perm0
        xyz_perm = xyz[:, perm, :]  # [1, N, 3]
        # 运行 FPS（不随机起点）
        _, idx_perm = sample_farthest_points(xyz_perm, K=K, random_start_point=False)
        # idx_perm 是针对于 perm 后坐标的索引，映射回原索引：
        sel = perm[idx_perm[0]]  # [K]
    else:
        # 无固定起点，直接运行
        _, idx = sample_farthest_points(xyz, K=K, random_start_point=False)
        sel = idx[0]  # [K]

    # 统一把可选附加信息搬到同 device/dtype 后再切片
    out_points = points[sel]  # [K, 6]，包含 one-hot 等附加通道

    out_colors = None
    out_semantics = None
    if colors is not None:
        out_colors = torch.as_tensor(colors, dtype=torch.float32, device=device)[sel]
    if semantics is not None:
        # 保持 int 类型
        sem = torch.as_tensor(semantics, device=device)
        if sem.dtype != torch.long:
            sem = sem.to(torch.long)
        out_semantics = sem[sel]

    # 与你原函数的返回风格保持一致
    if out_colors is not None and out_semantics is not None:
        return out_points.cpu().numpy(), out_colors.cpu().numpy(), out_semantics.cpu().numpy()
    elif out_colors is not None:
        return out_points.cpu().numpy(), out_colors.cpu().numpy()
    else:
        return out_points

def Transform_Push2Fixed_point_onehot(global_points_onehot: torch.Tensor,
                                  fixed_point: torch.Tensor,
                                  push_action: torch.Tensor) -> torch.Tensor:


    dev  = 'cuda'
    dtype = global_points_onehot.dtype

    fixed_point = fixed_point.to(device=dev, dtype=dtype)      # [2]
    push_action = push_action.to(device=dev, dtype=dtype)      # [7]

    xyz   = global_points_onehot[:, :3]                        # [N,3]
    roles = global_points_onehot[:, 3:]                        # [N,3]  (保持不变)

    t = push_action[:3]                                        # [tx, ty, tz]
    q = push_action[3:7].unsqueeze(0)                          # [1,4]
    R_push = _quat_to_rotmat_torch(q).squeeze(0)               # [3,3]

    push_pose = torch.eye(4, device=dev, dtype=dtype)
    push_pose[:3, :3] = R_push
    push_pose[:3,  3] = t

    tz = push_action[2]   
    tz = tz.unsqueeze(0)                                   # [1]
    trans_fixed = torch.cat([fixed_point, tz], dim=-1)         # [3] = [fx, fy, tz]

    R_fixed = torch.tensor([[0., -1.,  0.],
                            [-1.,  0.,  0.],
                            [0.,   0., -1.]], device=dev, dtype=dtype)
    fixed_pose = torch.eye(4, device=dev, dtype=dtype)
    fixed_pose[:3, :3] = R_fixed
    fixed_pose[:3,  3] = trans_fixed

    T_2fixed = fixed_pose @ torch.linalg.inv(push_pose)        # [4,4]

    N = xyz.shape[0]
    ones = torch.ones((N, 1), device=dev, dtype=dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)                      # [N,4]
    xyz_tf = (T_2fixed @ xyz_h.T).T[:, :3]                     # [N,3]

    global_points_onehot_ee = torch.cat([xyz_tf, roles], dim=1)  # [N,6]

    return global_points_onehot_ee

def pc_normalize_for_obj_onehot(pc: torch.Tensor,
                        centroid: torch.Tensor,
                        m: torch.Tensor):
    device, dtype = pc.device, pc.dtype
    N, C = pc.shape

    xyz   = pc[:, :3]                 # [N,3]
    extra = pc[:, 3:] if C > 3 else None  # [N,C-3]（包含 one-hot 等）

    xyz_c    = xyz - centroid         # 平移到原点

    # 数值稳定：半径过小时用 1.0 避免除零/NaN
    eps = torch.tensor(1e-12, device=device, dtype=dtype)
    scale = torch.where(m > eps, m, torch.ones((), device=device, dtype=dtype))

    xyz_n = xyz_c / scale             # 归一化

    pc_norm = torch.cat([xyz_n, extra], dim=1) if extra is not None else xyz_n

    return pc_norm

def pc_normalize_grasp_onehot(pc: torch.Tensor):

    if not torch.is_tensor(pc):
        pc = torch.as_tensor(pc, dtype=torch.float32)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"期望 [N,>=3]，收到 {tuple(pc.shape)}")

    device, dtype = pc.device, pc.dtype
    N, C = pc.shape

    xyz   = pc[:, :3]                 # [N,3]
    extra = pc[:, 3:] if C > 3 else None  # [N,C-3]（包含 one-hot 等）

    centroid = xyz.mean(dim=0)        # [3]
    xyz_c    = xyz - centroid         # 平移到原点

    # 最大半径（最远点欧氏距离）
    m = torch.linalg.norm(xyz_c, ord=2, dim=1).max()  # 标量张量
    # 数值稳定：半径过小时用 1.0 避免除零/NaN
    eps = torch.tensor(1e-12, device=device, dtype=dtype)
    scale = torch.where(m > eps, m, torch.ones((), device=device, dtype=dtype))

    xyz_n = xyz_c / scale             # 归一化

    pc_norm = torch.cat([xyz_n, extra], dim=1) if extra is not None else xyz_n

    return pc_norm, centroid, m

def _quat_to_rotmat_torch(q: torch.Tensor) -> torch.Tensor:
    """
    q: (..., 4)  in [qx, qy, qz, qw]  (SciPy的 from_quat 默认顺序)
    return: (..., 3, 3)
    """
    # 归一化
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
    qx, qy, qz, qw = q.unbind(-1)

    # 按标准公式构造旋转矩阵
    # 参考: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    r00 = 1 - 2*(yy + zz)
    r01 =     2*(xy - wz)
    r02 =     2*(xz + wy)

    r10 =     2*(xy + wz)
    r11 = 1 - 2*(xx + zz)
    r12 =     2*(yz - wx)

    r20 =     2*(xz - wy)
    r21 =     2*(yz + wx)
    r22 = 1 - 2*(xx + yy)

    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)
    return R

def any_point_in_expanded_obb(
    A: Union[o3d.geometry.PointCloud, np.ndarray],
    B: Union[o3d.geometry.PointCloud, np.ndarray],
    expand_by: float = 0.5
) -> bool:
    """
    判断点云 B 是否有点落在点云 A 的 OBB 放大后（中心不变）的包围盒内。
    参数:
        A: 点云A（Open3D PointCloud 或 (N,3) numpy 数组）
        B: 点云B（Open3D PointCloud 或 (M,3) numpy 数组）
        expand_by: 放大量，0.5 表示在各轴上扩大 50%（即 1.5×）
    返回:
        bool: 若 B 中至少一个点落入放大 OBB 内，返回 True，否则 False
    """

    # 1) 计算 A 的 OBB
    obb = A.get_oriented_bounding_box()

    # 2) 按各轴等比例放大 (中心不变)
    scale = 1.0 + float(expand_by)
    if scale <= 0:
        raise ValueError("放大倍数无效：1 + expand_by 必须大于 0。")
    obb_expanded = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, obb.extent.copy())
    obb_expanded.scale(scale, obb_expanded.center)  # 原地缩放

    # 3) 判断 B 是否有点落在放大后的 OBB 内
    ptsB = np.asarray(B.points)
    if ptsB.shape[0] == 0:
        return False
    ptsB_open3d = o3d.utility.Vector3dVector(ptsB)
    inside_idx = obb_expanded.get_point_indices_within_bounding_box(ptsB_open3d)
    return len(inside_idx) > 0

def compute_distance_GOC(pose, obj_center):
    """
    Distance between the object's center of mass and the gripping plane under horizontal mapping.
    Assuming the center is the center of mass.
    """
    p = np.asarray(pose, dtype=float)
    c = np.asarray(obj_center, dtype=float)
    diff_xy = (p - c)[..., :2]
    return np.linalg.norm(diff_xy, axis=-1)

# def compute_close_width(
#     points_world: np.ndarray,
#     grip_pos: np.ndarray,
#     pad_width: float,                 
#     clearance: float = 0.0,           
#     centerline_y_offset: float = 0.0, 
# ) -> float:
#     """
#     The final gripping width is determined 
#     by adding a threshold to the critical 
#     collision width between the gripper and the object being gripped.
#     """
#     P = TransformPCD2EndLink(points_world, grip_pos)  # (N,3)
#     pose_points, _ = grasp_pcd_bluenoise_like(n_target=170, oversample=2000, seed=55926)
#     sence_points = fuse_state_torch_v2(P, pose_points)
#     sence_points = sence_points.cpu().numpy()
#     # x, y, z = sence_points[:, 0], sence_points[:, 1], sence_points[:, 2]
#     P = P.cpu().numpy()
#     x, y, z = P[:, 0], P[:, 1], P[:, 2]

#     # #----compute gripper mapping points (need y axis toward pad).
#     half_pw = 0.5 * pad_width
#     mask_band = (x >= -half_pw) & (x <= half_pw)
#     y_band = y[mask_band] - centerline_y_offset
#     #----no limit on width.
#     if y_band.size == 0:
#         return float("inf")

#     #----find the farthest point.
#     y_pos = y_band[y_band > 0.0]
#     y_neg = y_band[y_band < 0.0]
#     y_pos_max = np.max(y_pos) if y_pos.size > 0 else np.inf
#     y_neg_min = np.min(y_neg) if y_neg.size > 0 else -np.inf

#     #----compute the farthest distance on both sides.
#     right_gap = y_pos_max if np.isfinite(y_pos_max) else np.inf
#     left_gap  = -y_neg_min if np.isfinite(y_neg_min) else np.inf
#     #----check if both sides is empty
#     if not np.isfinite(right_gap) and not np.isfinite(left_gap):
#         return float("inf")
#     #----if one side is empty, the width depend on another one.
#     gap_to_center = max(right_gap, left_gap) if np.isfinite(right_gap) and np.isfinite(left_gap) \
#                     else (right_gap if np.isfinite(right_gap) else left_gap)
#     #----set a clearance for safely grasp.
#     half_width = max(0.0, gap_to_center + clearance)

#     fig, ax = plt.subplots(figsize=(6, 6))

#     # # 所有点（x,y）
#     ax.scatter(x, y, s=2, alpha=0.2, label="all points")

#     # # 带内点
#     ax.scatter(x[mask_band], y[mask_band],
#                 s=4, alpha=0.7, label="band points")

#     # # pad 之间的 x 区间
#     ax.axvline(-half_pw, color="black", linestyle="--", linewidth=1)
#     ax.axvline( half_pw, color="black", linestyle="--", linewidth=1)
#     ax.text(0, ax.get_ylim()[1]*0.95,
#             f"pad_width = {pad_width:.3f}",
#             ha="center", va="top")

#     # # 中心线（考虑 y_offset）
#     ax.axhline(centerline_y_offset, color="gray", linestyle=":", linewidth=1,
#                 label="centerline + offset")

#     # # 画出计算出的半宽（相对中心线）
#     y_c = centerline_y_offset
#     ax.axhline(y_c + half_width, color="red", linestyle="-", linewidth=1.5,
#                 label=f"+half_width = {half_width:.3f}")
#     ax.axhline(y_c - half_width, color="red", linestyle="-", linewidth=1.5)
#     title = 'gripper width'
#     width = 2.0 * half_width
#     # # 把最远点标出来
#     if np.isfinite(y_pos_max):
#         ax.scatter(0, y_c + right_gap, color="green", s=40, marker="^",
#                     label=f"right_gap = {right_gap:.3f}")
#     if np.isfinite(left_gap):
#         ax.scatter(0, y_c - left_gap, color="blue", s=40, marker="v",
#                     label=f"left_gap = {left_gap:.3f}")

#     ax.set_xlabel("x (along pad width)")
#     ax.set_ylabel("y (between fingers)")
#     ax.set_aspect("equal", adjustable="box")
#     ax.legend(loc="best")
#     ax.set_title(title or f"close width = {width:.3f}")

#     plt.tight_layout()
#     plt.show()
#     return 2.0 * half_width

def compute_close_width(
    points_world: np.ndarray,
    grip_pos: np.ndarray,
    pad_width: float,                 
    clearance: float = 0.0,           
    centerline_y_offset: float = 0.0, 
    ext = 0.0
) -> float:
    """
    根据点云计算最小夹爪开合（沿 y 方向），并可视化：
    - pad 区域
    - 物体点云
    - 最大接触宽度
    - 点云中心到夹爪中心线 x=0 的距离
    """
    P = TransformPCD2EndLink(points_world, grip_pos)  # (N,3), gripper 坐标系
    pose_points, _ = grasp_pcd_bluenoise_like(extend=ext)
    sence_points, _ = fuse_state_torch_v3(P, pose_points)
    if len(sence_points) <= 50:
        return 0, False
    sence_points = sence_points.cpu().numpy()

    # P = points_world.cpu().numpy()
    x, y, z = sence_points[:, 0], sence_points[:, 1], sence_points[:, 2]

    half_pw = 0.5 * pad_width
    mask_band = (x >= -half_pw) & (x <= half_pw)

    y_band = y[mask_band] - centerline_y_offset

    if y_band.size == 0:
        return float("inf"), False

    y_pos = y_band[y_band > 0.0]
    y_neg = y_band[y_band < 0.0]
    y_pos_max = np.max(y_pos) if y_pos.size > 0 else np.inf
    y_neg_min = np.min(y_neg) if y_neg.size > 0 else -np.inf

    right_gap = y_pos_max if np.isfinite(y_pos_max) else np.inf
    left_gap  = -y_neg_min if np.isfinite(y_neg_min) else np.inf

    if not np.isfinite(right_gap) and not np.isfinite(left_gap):
        return float("inf"), False

    gap_to_center = (
        max(right_gap, left_gap)
        if np.isfinite(right_gap) and np.isfinite(left_gap)
        else (right_gap if np.isfinite(right_gap) else left_gap)
    )

    half_width = max(0.0, gap_to_center + clearance)
    width = 2.0 * half_width

    # -------- 4) 可视化
    # fig, ax = plt.subplots(figsize=(6, 6))

    # # 所有点
    # ax.scatter(x, y, s=2, alpha=0.2, label="all points")

    # # 带内点
    # ax.scatter(x_band, y_band_full, s=4, alpha=0.7, label="band points")

    # # pad 边界（沿 x）
    # ax.axvline(-half_pw, color="black", linestyle="--", linewidth=1)
    # ax.axvline( half_pw, color="black", linestyle="--", linewidth=1)
    # ax.text(0, ax.get_ylim()[1]*0.95,
    #         f"pad_width = {pad_width:.3f}",
    #         ha="center", va="top")

    # # y 方向中心线
    # y_c = centerline_y_offset
    # ax.axhline(y_c, color="gray", linestyle=":", linewidth=1,
    #            label="centerline + offset")

    # # 开合半宽（沿 y）
    # ax.axhline(y_c + half_width, color="red", linestyle="-", linewidth=1.5,
    #            label=f"+half_width = {half_width:.3f}")
    # ax.axhline(y_c - half_width, color="red", linestyle="-", linewidth=1.5)

    # # -------- 5) 夹爪中心线 x = 0 以及点云中心到该线的距离
    # # 夹爪中心线（过 (0,0)，垂直于 x 轴）
    # ax.axvline(0.0, color="purple", linestyle="-.", linewidth=1.5,
    #            label="gripper centerline (x=0)")

    # # 点云中心点
    # # ax.scatter(center_x, center_y, s=60, marker="*", color="purple",
    # #            label=f"cloud center ({center_x:.3f}, {center_y:.3f})")

    # # # 中心点到中心线的垂直距离线段（沿 x）
    # # ax.plot([0.0, center_x], [center_y, center_y],
    # #         color="purple", linewidth=2)

    # # # 在中点位置标注距离数值
    # # mid_x = 0.5 * center_x
    # # ax.text(mid_x,
    # #         center_y + 0.5 * (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
    # #         f"d = {center_dist:.3f}",
    # #         ha="center", va="bottom", color="purple")

    # ax.set_xlabel("x (along pad width)")
    # ax.set_ylabel("y (between fingers)")
    # ax.set_aspect("equal", adjustable="box")
    # ax.legend(loc="best")
    # ax.set_title(f"gripper width = {width:.3f}")
    # plt.tight_layout()
    # plt.show()

    # 如果后续还要用，可以把 center_dist 一并返回（例如改成 return width, center_dist）
    return width, True


def collision_dect(global_pc, obj_pc, grasp_pose):
    """
    Using the gripper pc to crop global pc to detect collison,
    if no point in gripper boxs, we think there is no collison occur.
    """

    pass

def convert_single_graspnet_grasp_to_env_pose(t_g: np.ndarray,
                                            R_g: np.ndarray,
                                            depth: float) -> np.ndarray:
        """
        将 GraspNet 的单个抓取 (t_g, R_g, depth) 转成环境里 grasp() 期望的 7D pose。

        输入:
            t_g   : (3,)  GraspNet 给出的 translation（抓取点，机械臂坐标系）
            R_g   : (3,3) GraspNet 的旋转矩阵，列向量为 {x_g, y_g, z_g}，x_g 为接近方向
            depth : float GraspNet 给出的深度 (沿 x_g 的距离)

        输出:
            grasp_pose: (7,) = [x, y, z, qx, qy, qz, qw]
                        坐标系已经对齐到你当前 grasp() 使用的 tip 坐标系，
                        可以直接传给 grasp(pose) 使用。
        """

        # ---------- 1) 把位置从“指尖接触点”移动到“抓取框架中心 / tip” ----------
        # GraspNet 中 depth 通常定义为沿接近方向 x_g 到 gripper center 的距离
        # t_tip = t_contact + depth * x_g
        x_g = R_g[:, 0]                       # approach 方向
        t_tip = t_g + depth * x_g             # tip 在机械臂坐标系下的位置

        # ---------- 2) 旋转矩阵从 GraspNet 坐标系 G 映射到你环境的 tip 坐标系 ----------

        R_tip = np.zeros((3, 3), dtype=np.float32)
        R_tip[:, 0] = R_g[:, 2]       # x_tip 对齐 z_g
        R_tip[:, 1] = -R_g[:, 1]      # y_tip 对齐 -y_g
        R_tip[:, 2] = R_g[:, 0]       # z_tip 对齐 x_g (approach)

        # ---------- 3) 修正左/右手系，确保 cross(x_tip, y_tip) ≈ z_tip ----------
        if np.linalg.norm(np.cross(R_tip[:, 0], R_tip[:, 1]) - R_tip[:, 2]) > 0.1:
            # 左手系: 把 x 轴取反，转成右手系
            R_tip[:, 0] = -R_tip[:, 0]

        # 再做一次正交化，稳一点
        u, _, vh = np.linalg.svd(R_tip)
        R_tip = u @ vh

        grasp_pose = np.eye(4)
        grasp_pose[:3, 3] = t_tip
        grasp_pose[:3, :3] = R_tip

        return grasp_pose

def compute_approach_angle(approach_vector):
    flag = True
    x_axis = approach_vector
    angle_rad = math.asin(abs(x_axis[2]) / np.linalg.norm(x_axis))
    if math.degrees(angle_rad) < 45:
        flag = False
        # print(f"\033[32m ------------------------------------------ \033[0m")
        # print(f"\033[32m approach angle = {math.degrees(angle_rad)} \033[0m")
        # print(f"\033[32m ------------------------------------------ \033[0m")
    return flag

def adjust_approach_angle(R: np.ndarray,
                          min_angle_deg: float = 30.0,
                          axis_index: int = 0) -> np.ndarray:
    """
    调整旋转矩阵中 approach 轴与水平面的夹角：
    - 若当前角度 < min_angle_deg，则把它调到恰好 min_angle_deg；
    - 若 >= min_angle_deg，则保持不变。
    
    R         : (3,3) 旋转矩阵
    min_angle_deg : 与 XY 平面的最小夹角（单位：度）
    axis_index    : 哪一列是 approach 轴（0/1/2），你现在是 [:3,0] 就用 0。
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)

    # 取出当前 approach 轴并单位化
    v = R[:, axis_index]
    v = v / (np.linalg.norm(v) + 1e-12)

    # 当前与 XY 平面的夹角（0° = 完全水平, 90° = 垂直）
    angle_rad = math.asin(abs(v[2]) / (np.linalg.norm(v) + 1e-12))
    angle_deg = math.degrees(angle_rad)
    print(f"\033[32m approach angle (before) = {angle_deg:.3f} deg \033[0m")

    if angle_deg >= min_angle_deg - 1e-6:
        # 已满足要求，直接返回
        return R.copy()

    # ------------ 构造与平面夹角 = min_angle_deg 的目标向量 v_new ------------
    target_rad = math.radians(min_angle_deg)

    # 保持 XY 投影方向不变
    v_xy = np.array([v[0], v[1], 0.0], dtype=float)
    norm_xy = np.linalg.norm(v_xy)
    if norm_xy < 1e-8:
        # 几乎垂直，理论上不会进入 angle < min_angle 的分支，保险起见原样返回
        return R.copy()

    dir_xy = v_xy / norm_xy                 # XY 平面中的方向
    v_new_xy = math.cos(target_rad) * dir_xy
    sign_z = 1.0 if v[2] >= 0.0 else -1.0   # 保持原来的“向上/向下”符号
    v_new_z = sign_z * math.sin(target_rad)

    v_new = np.array([v_new_xy[0], v_new_xy[1], v_new_z], dtype=float)
    v_new /= (np.linalg.norm(v_new) + 1e-12)

    # ------------ 计算把 v 旋到 v_new 的最小旋转 ΔR（Rodrigues）------------
    dot = float(np.clip(np.dot(v, v_new), -1.0, 1.0))
    if abs(dot - 1.0) < 1e-8:
        # 几乎已对齐
        return R.copy()

    angle = math.acos(dot)
    axis = np.cross(v, v_new)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        # 数值退化，直接返回
        return R.copy()
    axis = axis / axis_norm

    x, y, z = axis
    K = np.array([[0, -z,  y],
                  [z,  0, -x],
                  [-y, x,  0]])
    dR = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)

    # 对整个姿态做刚体旋转
    R_new = dR @ R

    # 检查新角度
    v_chk = R_new[:, axis_index]
    angle_new = math.degrees(
        math.asin(abs(v_chk[2]) / (np.linalg.norm(v_chk) + 1e-12))
    )
    print(f"\033[32m approach angle (after)  = {angle_new:.3f} deg \033[0m")
    print(f"\033[32m --------------------------------------------- \033[0m")
    return R_new

def compute_location_score(pose_translation, pcd, distance_force):
    
    obb, center, max_vertical_distance = analyze_point_cloud(pcd)
    pose_translation = np.array(pose_translation)
    center = np.array(center)
    distance_force = np.linalg.norm(pose_translation[:2] - center[:2])
    score = 1 - distance_force / max_vertical_distance
    # --- 可视化 ---
    # 创建一个坐标球代表质心
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
    center_sphere.paint_uniform_color([0, 0, 1]) # 蓝色
    center_sphere.translate(center)

    # 创建坐标轴
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=center)

    print("正在打开可视化窗口...")
    o3d.visualization.draw_geometries(
        [pcd, obb, center_sphere, mesh_frame], 
        window_name="包围框与质心分析",
        width=800, height=600
    )
    return score

def analyze_point_cloud(pcd):
    # ---------------------------------------------------------
    # 1. 估计点云中心 (质心)
    # ---------------------------------------------------------
    # get_center() 返回所有点的算术平均值
    center = pcd.get_center()
    print(f"1. 点云质心 (Center): {center}")
    # ---------------------------------------------------------
    # 4. (可选) 生成定向包围框 (OBB - Oriented Bounding Box)
    # ---------------------------------------------------------
    # OBB 是最小体积包围框，可以任意旋转
    obb = pcd.get_oriented_bounding_box()
    obb.color = (0, 1, 0)  # 绿色显示 OBB
    # OBB 的"垂直"是相对于物体自身的局部坐标系的
    # extent 返回 [width, height, depth]
    obb_extent = obb.extent
    max_length = max(obb_extent)
    print(f"6. OBB 局部尺寸 (Extent): {obb_extent}")
    print(f"   max_force_length: {max_length / 2:.4f}")

    return obb, center, max_length / 2

def visualize_center_distance_obb(points_world, grip_pos):
    """
    1) 将点云变换到末端坐标系：P = TransformPCD2EndLink(points_world, grip_pos)
    2) 用 Open3D 计算 P 的 OBB
    3) 取 OBB 中心点作为“点云中心”，计算并可视化：
         - D: OBB 中心到夹爪中心线 x=0 在 xy 平面的距离
         - d: OBB 中心到 OBB 某一对平面之一的最大垂直距离
            （沿最长轴方向的 1/2 extent）
    4) 在 xy 平面上画出：
         - 全部点云
         - 夹爪中心线 x=0
         - OBB 中心点
         - D 的线段
         - d 在 xy 平面的投影线段
    返回:
        D: float, OBB 中心到 x=0 直线在 xy 平面的距离
        center: np.ndarray, (3,), OBB 中心
        d_max_plane: float, OBB 中心到 OBB 框平面的最大垂直距离
    """
    # -------- 1) 变换到末端坐标系
    P = TransformPCD2EndLink(points_world, grip_pos)  # (N, 3)

    # 若是 torch.Tensor，转为 numpy
    if hasattr(P, "cpu"):
        P = P.cpu().numpy()
    else:
        P = np.asarray(P)

    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    # -------- 2) 计算 OBB
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)

    obb = pcd.get_oriented_bounding_box()
    center = np.asarray(obb.center)       # (3,)
    extent = np.asarray(obb.extent)       # (3,)  [ex, ey, ez]
    R = np.asarray(obb.R)                 # (3, 3)  每列为一个主轴方向

    cx, cy, cz = center

    # 夹爪中心线：过 (0,0)，垂直 X 轴 → x = 0
    D = float(abs(cx))   # 在 xy 平面的距离

    # -------- 3) 计算 d：中心到 OBB 某一对平面之一的最大垂直距离
    # 最长轴
    i_max = int(extent.argmax())
    # 该轴方向（单位向量）
    axis_dir = R[:, i_max]
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)

    # 中心到该轴对应平面的距离
    d_max_plane = 0.5 * float(extent[i_max])

    # 中心指向平面上的端点（3D）
    end_point_3d = center + axis_dir * d_max_plane  # (3,)

    # 在 xy 平面的投影
    end_x, end_y = end_point_3d[0], end_point_3d[1]

    # -------- 4) 可视化（xy 平面）
    fig, ax = plt.subplots(figsize=(6, 6))

    # 所有点
    ax.scatter(x, y, s=2, alpha=0.3, label="points")

    # 夹爪中心线 x = 0 （计算 D 用）
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1,
               label="gripper centerline (x=0)")

    # OBB 中心点
    ax.scatter(cx, cy, s=60, marker="*", color="red",
               label=f"OBB center ({cx:.3f}, {cy:.3f})")

    # D：中心到 x=0 的线段
    ax.plot([0.0, cx], [cy, cy], color="red", linewidth=2)
    mid_x_D = 0.5 * cx
    ax.text(mid_x_D, cy,
            f"D = {D:.3f}",
            ha="center", va="bottom", color="red")

    # d：中心到 OBB 最远平面的距离线段（在 xy 平面的投影）
    ax.plot([cx, end_x], [cy, end_y],
            color="orange", linewidth=2,
            label=f"d (to OBB face) = {d_max_plane:.3f}")

    # d 的文字标注放在线段中点
    mid_x_d = 0.5 * (cx + end_x)
    mid_y_d = 0.5 * (cy + end_y)
    ax.text(mid_x_d, mid_y_d,
            f"d = {d_max_plane:.3f}",
            ha="center", va="bottom", color="orange")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    ax.set_title(
        f"OBB center distance: D={D:.3f} (to x=0), "
        f"d={d_max_plane:.3f} (to OBB face)"
    )
    plt.tight_layout()
    plt.show()
    print(f'grasp score:{1 - D/d_max_plane}')
    return D, center, d_max_plane

def check_collision_voxel_then_kdtree(global_points_ee_tensor: torch.Tensor,
                                      pose_points_tensor: torch.Tensor,
                                      voxel_size_vox: float = 0.005,
                                      threshold: float = 0.003) -> bool:
    vox_env = _voxel_keys_from_tensor(global_points_ee_tensor, voxel_size_vox)
    vox_pose = _voxel_keys_from_tensor(pose_points_tensor, voxel_size_vox)
    vox_pose_dilated = _dilate_voxel_keys(vox_pose)

    if len(vox_env) == 0 or len(vox_pose_dilated) == 0 or len(vox_env.intersection(vox_pose_dilated)) == 0:
        return False

    env_np = global_points_ee_tensor.detach().cpu().numpy()
    pose_np = pose_points_tensor.detach().cpu().numpy()
    if env_np.shape[0] == 0 or pose_np.shape[0] == 0:
        return False

    tree = cKDTree(env_np)
    hits = tree.query_ball_point(pose_np, r=threshold)
    return any(len(h) > 0 for h in hits)

def _voxel_keys_from_tensor(points_tensor: torch.Tensor, voxel_size: float):
    if points_tensor.numel() == 0:
        return set()
    pts = points_tensor.detach().cpu().numpy()
    idx = np.floor(pts / voxel_size).astype(np.int64)
    return set(map(tuple, idx))

def _dilate_voxel_keys(voxel_keys: set):
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    out = set()
    for k in voxel_keys:
        i, j, kz = k
        for dx, dy, dz in offsets:
            out.add((i+dx, j+dy, kz+dz))
    return out

def refine_pose(pose, depth_i, rotation_j, depth_bin=0.008, rotation_bin=15):
    """
    refine pose according to depth bin and rotation bin.
    depth bin = 0.005 (1~3)
    rotation bin = 5 (-15~15)
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"pose must be a 4x4 homogeneous matrix, got {pose.shape}")

    depth_refine = float(depth_i) * float(depth_bin)                 
                                
    Tz = np.eye(4, dtype=np.float64)
    R = np.eye(4, dtype=np.float64)
    Tz[2, 3] = depth_refine   
    if rotation_j != 0:
        if rotation_j % 2 == 1:
            rotation_j = (rotation_j + 1) / 2
            rot_deg = float(rotation_j) * float(rotation_bin) # 1 3 5
        else:
            rotation_j = -rotation_j / 2
            rot_deg = float(rotation_j) * float(rotation_bin)    
        theta = np.deg2rad(rot_deg) 
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[ c, -s, 0.0],
                    [ s,  c, 0.0],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
        R[:3, :3] = Rz

    delta = Tz @ R
    refined_pose = pose @ delta
    return refined_pose, int(rotation_j)

def refine_pose_v2(pose, forward_j, depth_bin=0.01, forward_bin=0.015):
    """
    depth bin = 0.005 
    forward_bin = 0.005
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"pose must be a 4x4 homogeneous matrix, got {pose.shape}")
             
    Tz = np.eye(4, dtype=np.float64)
    Tx = np.eye(4, dtype=np.float64)
    # if forward_j == 0:
    depth_i = 1
    depth_refine = float(depth_i) * float(depth_bin)   
    Tz[2, 3] = depth_refine   
    if forward_j != 0:
        if forward_j % 2 == 1:
            forward_j = (forward_j + 1) / 2
            forward_deg = float(forward_j) * float(forward_bin) # 1 -1
            Tx[0, 3] = forward_deg
        else:
            forward_j = -forward_j / 2
            forward_deg = float(forward_j) * float(forward_bin)    
            Tx[0, 3] = forward_deg
    delta = Tz @ Tx
    refined_pose = pose @ delta
    return refined_pose, int(forward_j)

def crop_cloud_by_pose_matrix(
    points, 
    radius=0.10, 
    height=0.0475, 
    angle_deg=30.0,
    pose_matrix=None
):
    """
    输入:
        points: (N, 3) 原始点云
        pose_matrix: (4, 4) 齐次变换矩阵。
                     - Col 0 (X轴): 圆柱轴线 (顶面指向底面)
                     - Col 1 (Y轴): 平行于切线的直径
                     - Col 2 (Z轴): 垂直于切线的方向
                     - Col 3 (Origin): 顶面中心
        radius, height: 几何尺寸
        angle_deg: 切角
    输出:
        inside_points: 内部点云
        mask: 布尔掩码
    """
    
    # --- 1. 将全局点云转换到局部坐标系 ---
    # 局部坐标系定义: 原点在顶面中心，X朝底面，Y平行切线，Z垂直切线
    
    # 方法 A: 使用矩阵逆 (数学上最严谨)
    # T_local_to_global = pose_matrix
    # T_global_to_local = inv(pose_matrix)
    # Point_local = T_global_to_local * Point_global
    
    # 为了计算高效，我们可以手动提取旋转和平移 (利用旋转矩阵的正交性 R^-1 = R^T)
    pose_matrix = np.eye(4)
    R = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3]
    
    # 向量 P_vec = P_global - Origin
    diff_vec = points - t
    
    local_x = np.dot(diff_vec, R[:, 0]) # 沿 X轴 (高度方向) 
    local_y = np.dot(diff_vec, R[:, 1]) # 沿 Y轴 
    local_z = np.dot(diff_vec, R[:, 2]) # 沿 Z轴 

    mask_height = (local_x >= -height) & (local_x <= 0)

    mask_radius = (local_y**2 + local_z**2) <= radius**2 

    cut_half_width = radius * np.sin(np.deg2rad(angle_deg)) 
    mask_cut = np.abs(local_z) <= cut_half_width 

    final_mask = mask_height & mask_radius & mask_cut 
    return points[final_mask], final_mask

def visualize_result(points, mask, pose_matrix):
    """
    可视化函数: 显示世界坐标系、输入位姿坐标系、内部点(红)、外部点(灰)
    """
    print(f"总点数: {len(points)}")
    print(f"内部点数 (红色): {np.sum(mask)}")
    
    # 1. 内部点云 (红色)
    pcd_in = o3d.geometry.PointCloud()
    pcd_in.points = o3d.utility.Vector3dVector(points[mask])
    pcd_in.paint_uniform_color([1, 0, 0]) # Red
    
    # 2. 外部点云 (灰色, 半透明感)
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points[~mask])
    pcd_out.paint_uniform_color([0.8, 0.8, 0.8]) # Grey
    
    # 3. 世界坐标系 (原点)
    frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0,0,0])
    
    # 4. 输入位姿坐标系 (Pose Frame)
    # Open3D 的 create_coordinate_frame 默认是在原点，RGB对应XYZ
    # 直接应用 transform(pose_matrix) 即可把它移动到指定位姿
    frame_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
    frame_pose.transform(pose_matrix)
    
    print("可视化说明:")
    print("  - [大坐标轴] 世界原点")
    print("  - [悬浮坐标轴] 输入位姿:")
    print("      -> 红色 (X): 圆柱高度方向")
    print("      -> 绿色 (Y): 切线平行方向")
    print("      -> 蓝色 (Z): 切面宽度控制方向")
    
    o3d.visualization.draw_geometries(
        [pcd_out, pcd_in, frame_world, frame_pose],
        window_name="Matrix Pose Crop Check",
        width=1024, height=768
    )



def get_mask_bottom_origin_z_axis(local_points, radius, height, extend=0, angle_deg=30.0):
    """
    输入: 已经变换到局部坐标系的点云 (N, 3)
    
    坐标系严格定义:
      - Origin (0,0,0): 底面中心 (Bottom Center)
      - Z axis (+Z): 顶面 -> 底面 (Top -> Bottom)
        -> 推导: 底面是0，顶面是-Height。圆柱体位于 Z轴的 [-Height, 0] 区间。
      - Y axis: 平行于切线
      - X axis: 右手系生成的切面法线 (控制宽度)
    """
    x = local_points[:, 0]
    y = local_points[:, 1]
    z = local_points[:, 2]
    
    # --- 1. 高度约束 (Z轴) ---
    # 关键修正点: 有效区间是 [-Height, 0]
    mask_height = (z >= -height) & (z <= extend)
    
    # --- 2. 切面宽度约束 (X轴) ---
    # X轴是切面的法线，控制保留宽度
    cut_half_width = radius * np.sin(np.deg2rad(angle_deg))
    mask_cut = np.abs(x) <= cut_half_width
    
    # --- 3. 圆柱半径约束 (XY平面) ---
    # 截面是垂直于 Z轴 的 XY 平面
    mask_radius = (x**2 + y**2) <= radius**2
    
    # --- 4. 综合 ---
    final_mask = mask_height & mask_cut & mask_radius
    
    return final_mask, cut_half_width

def visualize_correction(local_points, radius=0.06, height=0.1,extend=0.008, angle_deg=15.0):
    # 1. 计算 Mask
    mask, cut_w = get_mask_bottom_origin_z_axis(local_points, radius, height, extend, angle_deg)
    
    print(f"输入点数: {len(local_points)}")
    print(f"内部点数: {np.sum(mask)}")
    
    # 2. 颜色区分
    pcd_in = o3d.geometry.PointCloud()
    pcd_in.points = o3d.utility.Vector3dVector(local_points[mask])
    pcd_in.paint_uniform_color([1, 0, 0]) # 红色 (内部)
    
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(local_points[~mask])
    pcd_out.paint_uniform_color([0.9, 0.9, 0.9]) # 灰色 (外部)
    
    # 3. 坐标轴 (原点 = 底面中心)
    # 红色=X, 绿色=Y, 蓝色=Z
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=radius, origin=[0,0,0])
    
    # 4. 绘制蓝色线框 (辅助验证)
    # 这里的关键是 Z 的坐标范围是 [-height, 0]
    
    # 定义8个顶点
    # Z=0 (底面), Z=-height (顶面)
    # X= +/- cut_w
    # Y= +/- radius (为了画框方便，画个外接矩形)
    
    box_corners = [
        [-cut_w, -radius, 0],       [-cut_w, radius, 0],       # Bottom (-X)
        [cut_w, -radius, 0],        [cut_w, radius, 0],        # Bottom (+X)
        [-cut_w, -radius, -height], [-cut_w, radius, -height], # Top (-X)
        [cut_w, -radius, -height],  [cut_w, radius, -height]   # Top (+X)
    ]
    
    # 连接线
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0], # 底面框 (Z=0)
        [4, 5], [5, 7], [7, 6], [6, 4], # 顶面框 (Z=-height)
        [0, 4], [1, 5], [2, 6], [3, 7]  # 连接棱
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(box_corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0, 0, 1]) # 蓝色
    
    print("\n可视化验证 (Bottom Origin + Z Axis Corrected):")
    print("1. [坐标原点]: 位于底面中心。")
    print("2. [蓝色轴 Z]: 指向 Top->Bottom 方向。")
    print("   -> 注意: 红色点云应该位于蓝色轴的**反方向** (负半轴)。")
    print("   -> 也就是 Z轴 像尾巴一样从红色点云的屁股后面伸出来。")
    print("3. [蓝色线框]: 完美包裹住红色点云，且 Z 范围是 [0 到 -H]。")
    
    o3d.visualization.draw_geometries(
        [pcd_in, pcd_out, frame, line_set],
        window_name="Corrected: Bottom Origin, Z-Axis",
        width=1024, height=768
    )


# def get_mask_bottom_origin_z_axis_torch(
#     local_points: torch.Tensor,   # [N,3] torch
#     radius: float,
#     height: float,
#     extend: float = 0.0,
#     angle_deg: float = 30.0,
# ):
#     """
#     输入: 已经变换到局部坐标系的点云 local_points (N,3), torch tensor
#     输出: mask (N,) torch.bool, cut_half_width float
#     """
#     x = local_points[:, 0]
#     y = local_points[:, 1]
#     z = local_points[:, 2]

#     # Z: [-height, extend]
#     mask_height = (z >= -height) & (z <= extend)

#     # X cut
#     cut_half_width = radius * math.sin(math.radians(angle_deg))
#     mask_cut = x.abs() <= cut_half_width

#     # radius in XY
#     r2 = radius * radius
#     mask_radius = (x * x + y * y) <= r2

#     return (mask_height & mask_cut & mask_radius), cut_half_width

def get_mask_bottom_origin_z_axis_torch(
    local_points: torch.Tensor,
    radius: float,
    height: float,
    extend: float = 0.0,
    angle_deg: float = 30.0,
):
    x = local_points[:, 0]
    y = local_points[:, 1]
    z = local_points[:, 2]

    sin_theta = math.sin(math.radians(angle_deg))
    cut_half_width = radius * sin_theta
    cut_half_width2 = cut_half_width * cut_half_width
    r2 = radius * radius

    x2 = x * x
    mask = (
        (z >= -height) &
        (z <= extend) &
        (x2 <= cut_half_width2) &
        ((x2 + y * y) <= r2)
    )
    return mask, cut_half_width


from functools import lru_cache

# 你已有的 gripper_point_width(n_target=200, oversample=..., gripper_width=...) 继续用
# 这里仅做缓存 + 降低 oversample（你可按精度需求调 600~1200）

@lru_cache(maxsize=512)
def _gripper_points_cached(width_mm: int):
    w = width_mm / 1000.0
    pts, _ = gripper_point_width(n_target=200, oversample=5000, gripper_width=w)  # oversample 下调
    # pts 通常在 CPU，缓存为 CPU tensor，使用时再搬到 GPU
    return pts.contiguous()

@lru_cache(maxsize=512)
def _gripper_obb_params_cached(width_mm: int):
    gripper_np = _gripper_points_cached(width_mm).detach().cpu().numpy()
    gripper_pc = o3d.geometry.PointCloud()
    gripper_pc.points = o3d.utility.Vector3dVector(gripper_np)
    _, gripper_obb = extend_obb_single_dir_along_global_z(gripper_pc)

    center = torch.from_numpy(np.asarray(gripper_obb.center, dtype=np.float32)).contiguous()
    axes = torch.from_numpy(np.asarray(gripper_obb.R, dtype=np.float32)).contiguous()
    half_extent = torch.from_numpy((np.asarray(gripper_obb.extent, dtype=np.float32) * 0.5)).contiguous()
    return center, axes, half_extent

def get_gripper_points(width: float, device: torch.device, dtype: torch.dtype):
    width_mm = int(round(float(width) * 1000.0))
    pts = _gripper_points_cached(width_mm)
    return pts.to(device=device, dtype=dtype, non_blocking=True)

def get_gripper_obb_params(width: float, device: torch.device, dtype: torch.dtype):
    width_mm = int(round(float(width) * 1000.0))
    center, axes, half_extent = _gripper_obb_params_cached(width_mm)
    return (
        center.to(device=device, dtype=dtype, non_blocking=True),
        axes.to(device=device, dtype=dtype, non_blocking=True),
        half_extent.to(device=device, dtype=dtype, non_blocking=True),
    )

def fuse_state_torch_obb_params(
    global_points: torch.Tensor,
    center: torch.Tensor,
    axes: torch.Tensor,
    half_extent: torch.Tensor,
    eps: float = 1e-6,
):
    local_points = (global_points - center) @ axes
    mask = (local_points.abs() <= (half_extent + eps)).all(dim=1)
    crop_indices = torch.nonzero(mask, as_tuple=False).flatten()
    return global_points[crop_indices], crop_indices

def fuse_state_torch_cached_obb(
    global_points: torch.Tensor,
    width: float,
    eps: float = 1e-6,
):
    center, axes, half_extent = get_gripper_obb_params(
        width,
        device=global_points.device,
        dtype=global_points.dtype,
    )
    return fuse_state_torch_obb_params(
        global_points,
        center=center,
        axes=axes,
        half_extent=half_extent,
        eps=eps,
    )
