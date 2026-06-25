# import argparse
# import numpy as np
# import random
# import datetime
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import open3d as o3d
# import os
# from tqdm import tqdm
# from refine.models.networks_v2 import Space_GraspFusion
# import refine.utils  as utils
# from scipy.spatial.transform import Rotation as R
# import time
# import math

# def orthogonalize_rot_batch(R: torch.Tensor) -> torch.Tensor:
#     """
#     R: [B,3,3]
#     通过 SVD 投影到 SO(3)
#     """
#     U, _, Vh = torch.linalg.svd(R)
#     R_ortho = U @ Vh

#     det = torch.linalg.det(R_ortho)
#     neg = det < 0
#     if neg.any():
#         U_fix = U.clone()
#         U_fix[neg, :, -1] *= -1
#         R_ortho = U_fix @ Vh

#     return R_ortho


# def get_mask_bottom_origin_z_axis_torch_batch(
#     local_points: torch.Tensor,   # [B, N, 3]
#     radius: torch.Tensor,         # [B]
#     height: float,
#     extend: float = 0.0,
#     angle_deg: float = 30.0,
# ):
#     """
#     batch 版本
#     输入:
#         local_points: [B, N, 3]
#         radius:       [B]
#     输出:
#         mask: [B, N] bool
#         cut_half_width: [B]
#     """
#     assert local_points.ndim == 3 and local_points.shape[-1] == 3, \
#         f"local_points 应为 [B,N,3]，实际 {local_points.shape}"
#     assert radius.ndim == 1 and radius.shape[0] == local_points.shape[0], \
#         f"radius 应为 [B]，实际 {radius.shape}"

#     x = local_points[:, :, 0]
#     y = local_points[:, :, 1]
#     z = local_points[:, :, 2]

#     mask_height = (z >= -height) & (z <= extend)

#     cut_half_width = radius * math.sin(math.radians(angle_deg))   # [B]
#     mask_cut = x.abs() <= cut_half_width[:, None]

#     r2 = radius * radius
#     mask_radius = (x * x + y * y) <= r2[:, None]

#     return (mask_height & mask_cut & mask_radius), cut_half_width


# class GraspEval():
#     def __init__(self, model_dir, seed, device):
#         # ---------- Arg Parser ----------
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         random.seed(seed)
#         self.device = device
#         self.model_dir = model_dir
#         self.model = Space_GraspFusion(device='cuda').to(device)
#         self.checkpoint = torch.load(self.model_dir, map_location='cuda')  # self.model_dir 为路径
#         self.model.load_state_dict(self.checkpoint['model'])

#     def evalueate_grasp_actions(self, global_sv_ply, poses):
#         grasp_trial_num = len(poses)
#         uniform_fuse_points = []
#         close_points_indexs = []
#         pose_excutes = []
#         widths = []
#         self.model.eval()
#         if grasp_trial_num < 1:
#             print('\033[32m No grasp pose be generated at current state! \033[0m')
#             return False, None, None
#         with torch.no_grad():
#             for i in range(grasp_trial_num):
#                 pos      = poses[i, 0:3]
#                 axis     = poses[i, 3:6]
#                 approach = poses[i, 6:9]
#                 binormal = poses[i, 9:12]
#                 width = poses[i, 12] 
#                 R_old = np.column_stack([-axis, binormal, approach])
            
#                 if not utils.compute_approach_angle(R_old[:3, 2]):
#                     # _, R_old = utils.adjust_pose_z_axis_to_down(R_old)
#                     # print('current grasp is down!!!')
#                     continue
#                 pose_excute = np.eye(4)
#                 pose_excute[:3, :3] = R_old
#                 pose_excute[:3, 3] = pos
#                 pose_excutes.append(pose_excute)
#                 widths.append(width)
#                 quaternion = R.from_matrix(pose_excute[:3 ,:3]).as_quat()
#                 record_pose = np.hstack([pose_excute[:3 ,3], quaternion])
#                 pose = torch.from_numpy(record_pose).float()
#                 sence_points = utils.TransformPCD2EndLink(global_sv_ply, pose)
#                 pose_points, _ = utils.gripper_point_width(n_target=200, gripper_width=width)
        
#                 crop_mask, _ = utils.get_mask_bottom_origin_z_axis(local_points=sence_points, radius=(width/2 + 0.0143), height=0.1, extend=0.008, angle_deg=30)
#                 # crop_mask = torch.from_numpy(crop_mask).to(device=sence_points.device) 
#                 crop_extend_points = sence_points[crop_mask]
#                 crop_extend_points_sample, _ = utils.furthest_point_sampling_nocuda(crop_extend_points, n_samples=720)
#                 gripper_close_pc, index = utils.fuse_state_torch_v3(crop_extend_points_sample, pose_points)
#                 if len(gripper_close_pc) < 50:
#                     continue
#                 _, sample_index = utils.furthest_point_sampling_nocuda(gripper_close_pc, n_samples=345)
#                 close_points_index = index[sample_index]
#                 uniform_fuse_point, _, _ = utils.pc_normalize_grasp(crop_extend_points_sample)
#                 uniform_fuse_point = uniform_fuse_point.T.to(dtype=torch.float32)
#                 uniform_fuse_points.append(uniform_fuse_point)
#                 close_points_indexs.append(close_points_index)
#             if len(uniform_fuse_points) == 0:
#                 return False, None, None
#             batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0) # Nx3 -> 10xNx3
#             batch_close_points_indexs = torch.stack(close_points_indexs, dim=0)
#             # batch_uniform_fuse_points = batch_uniform_fuse_points.transpose(1, 2) # 10xNx3 -> 10x3xN
#             batch_uniform_fuse_points = batch_uniform_fuse_points.to('cuda')
#             batch_close_points_indexs = batch_close_points_indexs.to('cuda')
#             pred = self.model(batch_uniform_fuse_points, batch_close_points_indexs)
#             pred_1 = pred['grasp_cls_pred']
#             pred_2 = pred['depth_rotation_cls_pred']
#             pred_1 = F.softmax(pred_1, dim=1)
#             pred_class = pred_1.data.max(1, keepdim=True)[1]
#         state_evaluate = False
#         action_idxs = (pred_class == 1).nonzero(as_tuple=True)[0]
#         if len(action_idxs) > 0:
#         # rank the succeessful actions and select the best action to execute
#             best_pre, best_action_idx = torch.max(pred_1[:,1], dim=0)
#             if best_pre >= 0.65:
#                 state_evaluate = True
#             grasp_pose = pose_excutes[best_action_idx]
#             print(f"\033[36m best_grasp_pre = {best_pre} \033[0m")
#             return state_evaluate, grasp_pose, widths[best_action_idx]
#         else:

#             print('\033[31m No valid grasp pose at that state! \033[0m')
#             return False, None, None

#     def evalueate_grasp_actions_return_bestpre(self, global_sv_ply, poses):
#         grasp_trial_num = len(poses)
#         uniform_fuse_points = []
#         close_points_indexs = []
#         pose_excutes = []
#         widths = []
#         self.model.eval()
#         if grasp_trial_num < 1:
#             print('\033[32m No grasp pose be generated at current state! \033[0m')
#             return False, None, None, None
#         with torch.no_grad():
#             for i in range(grasp_trial_num):
#                 pos      = poses[i, 0:3]
#                 axis     = poses[i, 3:6]
#                 approach = poses[i, 6:9]
#                 binormal = poses[i, 9:12]
#                 width = poses[i, 12] 
#                 R_old = np.column_stack([-axis, binormal, approach])
            
#                 if not utils.compute_approach_angle(R_old[:3, 2]):
#                     # _, R_old = utils.adjust_pose_z_axis_to_down(R_old)
#                     # print('current grasp is down!!!')
#                     continue
#                 pose_excute = np.eye(4)
#                 pose_excute[:3, :3] = R_old
#                 pose_excute[:3, 3] = pos
#                 pose_excutes.append(pose_excute)
#                 widths.append(width)
#                 quaternion = R.from_matrix(pose_excute[:3 ,:3]).as_quat()
#                 record_pose = np.hstack([pose_excute[:3 ,3], quaternion])
#                 pose = torch.from_numpy(record_pose).float()
#                 sence_points = utils.TransformPCD2EndLink(global_sv_ply, pose)
#                 pose_points, _ = utils.gripper_point_width(n_target=200, gripper_width=width)
        
#                 crop_mask, _ = utils.get_mask_bottom_origin_z_axis(local_points=sence_points, radius=(width/2 + 0.0143), height=0.1, extend=0.008, angle_deg=30)
#                 # crop_mask = torch.from_numpy(crop_mask).to(device=sence_points.device) 
#                 crop_extend_points = sence_points[crop_mask]
#                 crop_extend_points_sample, _ = utils.furthest_point_sampling_nocuda(crop_extend_points, n_samples=960)
#                 gripper_close_pc, index = utils.fuse_state_torch_v3(crop_extend_points_sample, pose_points)
#                 if len(gripper_close_pc) < 50:
#                     continue
#                 _, sample_index = utils.furthest_point_sampling_nocuda(gripper_close_pc, n_samples=345)
#                 close_points_index = index[sample_index]
#                 uniform_fuse_point, _, _ = utils.pc_normalize_grasp(crop_extend_points_sample)
#                 uniform_fuse_point = uniform_fuse_point.T.to(dtype=torch.float32)
#                 uniform_fuse_points.append(uniform_fuse_point)
#                 close_points_indexs.append(close_points_index)
#             if len(uniform_fuse_points) == 0:
#                 return False, None, None, None
#             batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0) # Nx3 -> 10xNx3
#             batch_close_points_indexs = torch.stack(close_points_indexs, dim=0)
#             # batch_uniform_fuse_points = batch_uniform_fuse_points.transpose(1, 2) # 10xNx3 -> 10x3xN
#             batch_uniform_fuse_points = batch_uniform_fuse_points.to('cuda')
#             batch_close_points_indexs = batch_close_points_indexs.to('cuda')
#             pred = self.model(batch_uniform_fuse_points, batch_close_points_indexs)
#             pred_1 = pred['grasp_cls_pred']
#             pred_2 = pred['depth_rotation_cls_pred']
#             pred_1 = F.softmax(pred_1, dim=1)
#             pred_class = pred_1.data.max(1, keepdim=True)[1]
#         state_evaluate = False
#         action_idxs = (pred_class == 1).nonzero(as_tuple=True)[0]
#         if len(action_idxs) > 0:
#         # rank the succeessful actions and select the best action to execute
#             best_pre, best_action_idx = torch.max(pred_1[:,1], dim=0)
#             if best_pre >= 0.65:
#                 state_evaluate = True
#             grasp_pose = pose_excutes[best_action_idx]
#             print(f"\033[36m best_grasp_pre = {best_pre} \033[0m")
#             return state_evaluate, grasp_pose, widths[best_action_idx], best_pre
#         else:

#             print('\033[31m No valid grasp pose at that state! \033[0m')
#             return False, None, None, None
    


#     # def evalueate_grasp_actions_fast(self, global_sv_ply, poses):
#     #     def sync():
#     #         if torch.cuda.is_available():
#     #             torch.cuda.synchronize()
            

#     #     t0 = time.time()
#     #     grasp_trial_num = len(poses)
#     #     uniform_fuse_points = []
#     #     close_points_indexs = []
#     #     pose_excutes = []
#     #     widths = []
#     #     self.model.eval()
#     #     if grasp_trial_num < 1:
#     #         print('\033[32m No grasp pose be generated at current state! \033[0m')
#     #         return False, None, None, None, None, None
#     #     t1 = time.time()
#     #     print(f'T0:{t1-t0}')
#     #     # with torch.no_grad():
#     #     with torch.inference_mode():
#     #         for i in range(grasp_trial_num):
#     #             sync()
#     #             T0 = time.perf_counter()
#     #             pos      = poses[i, 0:3]
#     #             axis     = poses[i, 3:6]
#     #             approach = poses[i, 6:9]
#     #             binormal = poses[i, 9:12]
#     #             width = poses[i, 12] 
#     #             R_old = np.column_stack([-axis, binormal, approach])
            
#     #             if not utils.compute_approach_angle(R_old[:3, 2]):
#     #                 # _, R_old = utils.adjust_pose_z_axis_to_down(R_old)
#     #                 # print('current grasp is down!!!')
#     #                 continue
#     #             sync()
#     #             T1 = time.perf_counter()
#     #             print(f't1:{T1-T0}')
#     #             pose_excute = np.eye(4)
#     #             pose_excute[:3, :3] = R_old
#     #             pose_excute[:3, 3] = pos
#     #             pose_excutes.append(pose_excute)
#     #             widths.append(width)
#     #             t = torch.as_tensor(pos, device=global_sv_ply.device, dtype=global_sv_ply.dtype)          # [3]
#     #             R = torch.as_tensor(R_old, device=global_sv_ply.device, dtype=global_sv_ply.dtype)

#     #             # SVD 正交化：把近似旋转投影为合法旋转
#     #             U, _, Vh = torch.linalg.svd(R)
#     #             R = U @ Vh
#     #             # 修正 det = +1（避免反射）
#     #             if torch.linalg.det(R) < 0:
#     #                 U[:, -1] *= -1
#     #                 R = U @ Vh

#     #             sence_points = (global_sv_ply - t.unsqueeze(0)) @ R
#     #             pose_points = utils.get_gripper_points(width, device=global_sv_ply.device, dtype=global_sv_ply.dtype)  # [200,3]
#     #             radius = float(width / 2.0 + 0.0143)
#     #             crop_mask, _ = utils.get_mask_bottom_origin_z_axis_torch(
#     #                 sence_points, radius=radius, height=0.1, extend=0.008, angle_deg=30.0
#     #             )
#     #             crop_extend_points = sence_points[crop_mask]
#     #             sync()
#     #             T2 = time.perf_counter()
#     #             print(f't2:{T2-T1}')

#     #             crop_extend_points_sample, _ = utils.fps_p3d(crop_extend_points, n_samples=960)
#     #             sync()
#     #             T3 = time.perf_counter()
#     #             print(f't3:{T3-T2}')

#     #             # print(f'sample time1:{T1-T0}')
#     #             gripper_close_pc, index = utils.fuse_state_torch_v3(crop_extend_points_sample, pose_points)
#     #             if len(gripper_close_pc) < 50:
#     #                 continue
#     #             sync()
#     #             T4 = time.perf_counter()
#     #             print(f't4:{T4-T3}')

#     #             _, sample_index = utils.fps_p3d(gripper_close_pc, n_samples=345)
#     #             sync()
#     #             T5 = time.perf_counter()
#     #             print(f't5:{T5-T4}')

#     #             # print(f'sample time1:{T3-T2}')
#     #             close_points_index = index[sample_index]
#     #             uniform_fuse_point, _, _ = utils.pc_normalize_grasp(crop_extend_points_sample)
#     #             uniform_fuse_point = uniform_fuse_point.T.to(dtype=torch.float32)
#     #             uniform_fuse_points.append(uniform_fuse_point)
#     #             close_points_indexs.append(close_points_index)
#     #             sync()
#     #             T6 = time.perf_counter()
#     #             print(f't6:{T6-T5}')
#     #             print(f't7:{T6-T0}')


#     #         if len(uniform_fuse_points) == 0:
#     #             return False, None, None, None, None, None
#     #         t2 = time.time()
#     #         print(f"T1:{t2-t1}")
#     #         # batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0) # Nx3 -> 10xNx3
#     #         # batch_close_points_indexs = torch.stack(close_points_indexs, dim=0)
#     #         batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0).contiguous().to(self.device, non_blocking=True)
#     #         batch_close_points_indexs = torch.stack(close_points_indexs, dim=0).contiguous().to(self.device, non_blocking=True)
#     #         # torch.backends.cudnn.benchmark = False
#     #         # batch_uniform_fuse_points = batch_uniform_fuse_points.transpose(1, 2) # 10xNx3 -> 10x3xN
#     #         starter = torch.cuda.Event(enable_timing=True)
#     #         ender   = torch.cuda.Event(enable_timing=True)
#     #         torch.cuda.synchronize()
#     #         starter.record()
#     #         pred = self.model(batch_uniform_fuse_points, batch_close_points_indexs)
#     #         ender.record()
#     #         torch.cuda.synchronize()
#     #         print(f"\033[35m model infernce time cost is {starter.elapsed_time(ender) / 1000.0} \033[0m")
#     #         pred_1 = pred['grasp_cls_pred']
#     #         pred_2 = pred['depth_rotation_cls_pred']
#     #         flat_idx = torch.argmax(pred_2)     
#     #         refin_row = (flat_idx // pred_2.size(1)).item()   
#     #         refine_col = (flat_idx %  pred_2.size(1)).item()
#     #         refine_val = pred_2[refin_row, refine_col].item()
#     #         refine_val_prob = torch.sigmoid(torch.tensor(refine_val))
#     #         pred_1 = F.softmax(pred_1, dim=1)
#     #         pred_class = pred_1.data.max(1, keepdim=True)[1]
#     #         t3 = time.time()
#     #         # print(f'T2:{t3-t2}')
#     #     refine = False
#     #     state_evaluate = False
#     #     # action_idxs = (pred_class == 1).nonzero(as_tuple=True)[0]
#     #     # if len(action_idxs) > 0:
#     #     # rank the succeessful actions and select the best action to execute
#     #     best_pre, best_action_idx = torch.max(pred_1[:,1], dim=0)
#     #     if best_pre > 0.6:
#     #         state_evaluate = True
#     #     elif refine_val_prob > 0.6:
#     #         print(f"\033[36m Enter refine mode!!! \033[0m")
#     #         state_evaluate = True
#     #         refine = True
#     #         best_action_idx = refin_row

#     #     grasp_pose = pose_excutes[best_action_idx]
#     #     print(f"\033[36m best_grasp_pre = {best_pre} \033[0m")
#     #     print(f"\033[36m best_refine_val = {refine_val_prob} \033[0m")
#     #     return state_evaluate, grasp_pose, widths[best_action_idx], best_pre, refine, refine_col
#     #     # else:
#     #     #     print('\033[31m No valid grasp pose at that state! \033[0m')
#     #     #     return False, None, None, None, None, None


#     def evalueate_grasp_actions_fast(self, global_sv_ply, poses):
#         def sync():
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
            

#         t0 = time.time()
#         grasp_trial_num = len(poses)
#         uniform_fuse_points = []
#         close_points_indexs = []
#         pose_excutes = []
#         widths = []
#         self.model.eval()
#         if grasp_trial_num < 1:
#             print('\033[32m No grasp pose be generated at current state! \033[0m')
#             return False, None, None, None, None, None
#         t1 = time.time()
#         # print(f'T0:{t1-t0}')
#         # with torch.no_grad():
#         with torch.inference_mode():
#             for i in range(grasp_trial_num):

#                 pos      = poses[i, 0:3]
#                 axis     = poses[i, 3:6]
#                 approach = poses[i, 6:9]
#                 binormal = poses[i, 9:12]
#                 width = poses[i, 12] 
#                 R_old = np.column_stack([-axis, binormal, approach])
            
#                 if not utils.compute_approach_angle(R_old[:3, 2]):
#                     # _, R_old = utils.adjust_pose_z_axis_to_down(R_old)
#                     # print('current grasp is down!!!')
#                     continue

#                 pose_excute = np.eye(4)
#                 pose_excute[:3, :3] = R_old
#                 pose_excute[:3, 3] = pos
#                 pose_excutes.append(pose_excute)
#                 widths.append(width)

#                 t = torch.as_tensor(pos, device=global_sv_ply.device, dtype=global_sv_ply.dtype)
#                 R = torch.as_tensor(R_old, device=global_sv_ply.device, dtype=global_sv_ply.dtype)

#                 U, _, Vh = torch.linalg.svd(R)
#                 R = U @ Vh
#                 if torch.linalg.det(R) < 0:
#                     U[:, -1] *= -1
#                     R = U @ Vh

#                 sence_points = (global_sv_ply - t.unsqueeze(0)) @ R

#                 pose_points = utils.get_gripper_points(width, device=global_sv_ply.device, dtype=global_sv_ply.dtype)
#                 radius = float(width / 2.0 + 0.0143)
#                 crop_mask, _ = utils.get_mask_bottom_origin_z_axis_torch(
#                     sence_points, radius=radius, height=0.1, extend=0.008, angle_deg=30.0
#                 )

#                 crop_extend_points = sence_points[crop_mask]


#                 crop_points_for_fps = crop_extend_points
#                 n_before = crop_points_for_fps.shape[0]

#                 crop_points_for_fps, _, used_voxel = utils.pre_voxel_downsample_for_fps(
#                     crop_points_for_fps,
#                     cap=1050,
#                     trigger=1200,
#                     voxel_size0=0.002,
#                     growth=1.35,
#                     max_iter=6,
#                 )

#                 n_after = crop_points_for_fps.shape[0]

#                 crop_extend_points_sample, _ = utils.fps_p3d(
#                     crop_points_for_fps,
#                     n_samples=960
#                 )


#                 # crop_extend_points_sample, _ = utils.fps_p3d(crop_extend_points, n_samples=960)

#                 gripper_close_pc, index = utils.fuse_state_torch_v3(crop_extend_points_sample, pose_points)
#                 if len(gripper_close_pc) < 50:
#                     continue

#                 _, sample_index = utils.fps_p3d(gripper_close_pc, n_samples=345)

#                 close_points_index = index[sample_index]
#                 uniform_fuse_point, _, _ = utils.pc_normalize_grasp(crop_extend_points_sample)
#                 uniform_fuse_point = uniform_fuse_point.T.to(dtype=torch.float32)
#                 uniform_fuse_points.append(uniform_fuse_point)
#                 close_points_indexs.append(close_points_index)

#             if len(uniform_fuse_points) == 0:
#                 return False, None, None, None, None, None
#             t2 = time.time()
#             print(f"Tpre:{t2-t1}")
#             # batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0) # Nx3 -> 10xNx3
#             # batch_close_points_indexs = torch.stack(close_points_indexs, dim=0)
#             batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0).contiguous().to(self.device, non_blocking=True)
#             batch_close_points_indexs = torch.stack(close_points_indexs, dim=0).contiguous().to(self.device, non_blocking=True)
#             # torch.backends.cudnn.benchmark = False
#             # batch_uniform_fuse_points = batch_uniform_fuse_points.transpose(1, 2) # 10xNx3 -> 10x3xN
#             starter = torch.cuda.Event(enable_timing=True)
#             ender   = torch.cuda.Event(enable_timing=True)
#             torch.cuda.synchronize()
#             starter.record()
#             pred = self.model(batch_uniform_fuse_points, batch_close_points_indexs)
#             ender.record()
#             torch.cuda.synchronize()
#             print(f"\033[35m model infernce time cost is {starter.elapsed_time(ender) / 1000.0} \033[0m")
#             pred_1 = pred['grasp_cls_pred']
#             pred_2 = pred['depth_rotation_cls_pred']
#             flat_idx = torch.argmax(pred_2)     
#             refin_row = (flat_idx // pred_2.size(1)).item()   
#             refine_col = (flat_idx %  pred_2.size(1)).item()
#             refine_val = pred_2[refin_row, refine_col].item()
#             refine_val_prob = torch.sigmoid(torch.tensor(refine_val))
#             pred_1 = F.softmax(pred_1, dim=1)
#             pred_class = pred_1.data.max(1, keepdim=True)[1]
#             t3 = time.time()
#             # print(f'T2:{t3-t2}')
#         refine = False
#         state_evaluate = False
#         # action_idxs = (pred_class == 1).nonzero(as_tuple=True)[0]
#         # if len(action_idxs) > 0:
#         # rank the succeessful actions and select the best action to execute
#         best_pre, best_action_idx = torch.max(pred_1[:,1], dim=0)
#         if best_pre > 0.6:
#             state_evaluate = True
#         elif refine_val_prob > 0.6:
#             print(f"\033[36m Enter refine mode!!! \033[0m")
#             state_evaluate = True
#             refine = True
#             best_action_idx = refin_row

#         grasp_pose = pose_excutes[best_action_idx]
#         print(f"\033[36m best_grasp_pre = {best_pre} \033[0m")
#         print(f"\033[36m best_refine_val = {refine_val_prob} \033[0m")
#         return state_evaluate, grasp_pose, widths[best_action_idx], best_pre, refine, refine_col
#         # else:
#         #     print('\033[31m No valid grasp pose at that state! \033[0m')
#         #     return False, None, None, None, None, None

#     def evalueate_grasp_actions_fast_v2(self, global_sv_ply, poses):
#         t0 = time.time()
#         grasp_trial_num = len(poses)
#         uniform_fuse_points = []
#         close_points_indexs = []
#         pose_excutes = []
#         widths = []
#         self.model.eval()
#         if grasp_trial_num < 1:
#             print('\033[32m No grasp pose be generated at current state! \033[0m')
#             return False, None, None, None, None, None
#         t1 = time.time()
#         # print(f'T0:{t1-t0}')
#         # with torch.no_grad():
#         with torch.inference_mode():
#             for i in range(grasp_trial_num):
#                 T0 = time.time()
#                 pos      = poses[i, 0:3]
#                 axis     = poses[i, 3:6]
#                 approach = poses[i, 6:9]
#                 binormal = poses[i, 9:12]
#                 width = poses[i, 12] 
#                 R_old = np.column_stack([-axis, binormal, approach])
            
#                 if not utils.compute_approach_angle(R_old[:3, 2]):
#                     # _, R_old = utils.adjust_pose_z_axis_to_down(R_old)
#                     # print('current grasp is down!!!')
#                     continue
#                 T1 = time.time()
#                 # print(f't1:{T1-T0}')
#                 pose_excute = np.eye(4)
#                 pose_excute[:3, :3] = R_old
#                 pose_excute[:3, 3] = pos
#                 pose_excutes.append(pose_excute)
#                 widths.append(width)
#                 t = torch.as_tensor(pos, device=global_sv_ply.device, dtype=global_sv_ply.dtype)          # [3]
#                 R = torch.as_tensor(R_old, device=global_sv_ply.device, dtype=global_sv_ply.dtype)

#                 # SVD 正交化：把近似旋转投影为合法旋转
#                 U, _, Vh = torch.linalg.svd(R)
#                 R = U @ Vh
#                 # 修正 det = +1（避免反射）
#                 if torch.linalg.det(R) < 0:
#                     U[:, -1] *= -1
#                     R = U @ Vh

#                 sence_points = (global_sv_ply - t.unsqueeze(0)) @ R
#                 pose_points = utils.get_gripper_points(width, device=global_sv_ply.device, dtype=global_sv_ply.dtype)  # [200,3]
#                 radius = float(width / 2.0 + 0.0143)
#                 crop_mask, _ = utils.get_mask_bottom_origin_z_axis_torch(
#                     sence_points, radius=radius, height=0.1, extend=0.008, angle_deg=30.0
#                 )
#                 crop_extend_points = sence_points[crop_mask]
#                 if crop_extend_points is None or crop_extend_points.shape[0] == 0:
#                     continue
#                 T2 = time.time()
#                 # print(f't2:{T2-T1}')

#                 crop_extend_points_sample, _ = utils.furthest_point_sampling_nocuda(crop_extend_points, n_samples=960)

#                 T3 = time.time()
#                 # print(f't3:{T3-T2}')

#                 # print(f'sample time1:{T1-T0}')
#                 gripper_close_pc, index = utils.fuse_state_torch_v3(crop_extend_points_sample, pose_points)
#                 if len(gripper_close_pc) < 50:
#                     continue
#                 T4 = time.time()
#                 # print(f't4:{T4-T3}')

#                 _, sample_index = utils.furthest_point_sampling_nocuda(gripper_close_pc, n_samples=345)
#                 T5 = time.time()
#                 # print(f't5:{T5-T4}')

#                 # print(f'sample time1:{T3-T2}')
#                 close_points_index = index[sample_index]
#                 uniform_fuse_point, _, _ = utils.pc_normalize_grasp(crop_extend_points_sample)
#                 uniform_fuse_point = uniform_fuse_point.T.to(dtype=torch.float32)
#                 uniform_fuse_points.append(uniform_fuse_point)
#                 close_points_indexs.append(close_points_index)
#                 T6 = time.time()
#                 # print(f't6:{T6-T5}')
#                 # print(f't7:{T6-T0}')


#             if len(uniform_fuse_points) == 0:
#                 return False, None, None, None, None, None
#             t2 = time.time()
#             # print(f"T1:{t2-t1}")
#             # batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0) # Nx3 -> 10xNx3
#             # batch_close_points_indexs = torch.stack(close_points_indexs, dim=0)
#             batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0).contiguous().to(self.device, non_blocking=True)
#             batch_close_points_indexs = torch.stack(close_points_indexs, dim=0).contiguous().to(self.device, non_blocking=True)
#             # torch.backends.cudnn.benchmark = False
#             # batch_uniform_fuse_points = batch_uniform_fuse_points.transpose(1, 2) # 10xNx3 -> 10x3xN
#             # starter = torch.cuda.Event(enable_timing=True)
#             # ender   = torch.cuda.Event(enable_timing=True)
#             # torch.cuda.synchronize()
#             # starter.record()
#             pred = self.model(batch_uniform_fuse_points, batch_close_points_indexs)
#             # ender.record()
#             # torch.cuda.synchronize()
#             # print(f"\033[35m model infernce time cost is {starter.elapsed_time(ender) / 1000.0} \033[0m")
#             pred_1 = pred['grasp_cls_pred']
#             pred_2 = pred['depth_rotation_cls_pred']
#             flat_idx = torch.argmax(pred_2)     
#             refin_row = (flat_idx // pred_2.size(1)).item()   
#             refine_col = (flat_idx %  pred_2.size(1)).item()
#             refine_val = pred_2[refin_row, refine_col].item()
#             refine_val_prob = torch.sigmoid(torch.tensor(refine_val))
#             pred_1 = F.softmax(pred_1, dim=1)
#             pred_class = pred_1.data.max(1, keepdim=True)[1]
#             t3 = time.time()
#             # print(f'T2:{t3-t2}')
#         refine = False
#         state_evaluate = False
#         # action_idxs = (pred_class == 1).nonzero(as_tuple=True)[0]
#         # if len(action_idxs) > 0:
#         # rank the succeessful actions and select the best action to execute
#         best_pre, best_action_idx = torch.max(pred_1[:,1], dim=0)
#         if best_pre > 0.8:
#             state_evaluate = True
#         elif refine_val_prob > 0.8:
#             print(f"\033[36m Enter refine mode!!! \033[0m")
#             state_evaluate = True
#             refine = True
#             best_action_idx = refin_row

#         grasp_pose = pose_excutes[best_action_idx]
#         print(f"\033[36m best_grasp_pre = {best_pre} \033[0m")
#         print(f"\033[36m best_refine_val = {refine_val_prob} \033[0m")
#         return state_evaluate, grasp_pose, widths[best_action_idx], best_pre, refine, refine_col
#         # else:
#         #     print('\033[31m No valid grasp pose at that state! \033[0m')
#         #     return False, None, None, None, None, None




    
import argparse
import numpy as np
import random
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import os
from tqdm import tqdm
from refine.models.networks_v2 import Space_GraspFusion
import refine.utils  as utils
from scipy.spatial.transform import Rotation as R
import time
import math

def orthogonalize_rot_batch(R: torch.Tensor) -> torch.Tensor:
    """
    R: [B,3,3]
    通过 SVD 投影到 SO(3)
    """
    U, _, Vh = torch.linalg.svd(R)
    R_ortho = U @ Vh

    det = torch.linalg.det(R_ortho)
    neg = det < 0
    if neg.any():
        U_fix = U.clone()
        U_fix[neg, :, -1] *= -1
        R_ortho = U_fix @ Vh

    return R_ortho


def get_mask_bottom_origin_z_axis_torch_batch(
    local_points: torch.Tensor,   # [B, N, 3]
    radius: torch.Tensor,         # [B]
    height: float,
    extend: float = 0.0,
    angle_deg: float = 30.0,
):
    """
    batch 版本
    输入:
        local_points: [B, N, 3]
        radius:       [B]
    输出:
        mask: [B, N] bool
        cut_half_width: [B]
    """
    assert local_points.ndim == 3 and local_points.shape[-1] == 3, \
        f"local_points 应为 [B,N,3]，实际 {local_points.shape}"
    assert radius.ndim == 1 and radius.shape[0] == local_points.shape[0], \
        f"radius 应为 [B]，实际 {radius.shape}"

    x = local_points[:, :, 0]
    y = local_points[:, :, 1]
    z = local_points[:, :, 2]

    mask_height = (z >= -height) & (z <= extend)

    cut_half_width = radius * math.sin(math.radians(angle_deg))   # [B]
    mask_cut = x.abs() <= cut_half_width[:, None]

    r2 = radius * radius
    mask_radius = (x * x + y * y) <= r2[:, None]

    return (mask_height & mask_cut & mask_radius), cut_half_width

def pc_normalize_grasp_batch(pc: torch.Tensor):
    """
    pc: [B, N, 3]
    返回:
        normalized: [B, N, 3]
        centroid  : [B, 3]
        radius    : [B]
    """
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc_centered = pc - centroid
    radius = torch.linalg.norm(pc_centered, dim=2).amax(dim=1).clamp_min(1e-12)
    normalized = pc_centered / radius[:, None, None]
    return normalized, centroid.squeeze(1), radius


class GraspEval():
    def __init__(self, model_dir, seed, device):
        # ---------- Arg Parser ----------
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.device = device
        self.model_dir = model_dir
        self.model = Space_GraspFusion(device='cuda').to(device)
        self.checkpoint = torch.load(self.model_dir, map_location='cuda')  # self.model_dir 为路径
        self.model.load_state_dict(self.checkpoint['model'])

    def evalueate_grasp_actions(self, global_sv_ply, poses):
        grasp_trial_num = len(poses)
        uniform_fuse_points = []
        close_points_indexs = []
        pose_excutes = []
        widths = []
        self.model.eval()
        if grasp_trial_num < 1:
            print('\033[32m No grasp pose be generated at current state! \033[0m')
            return False, None, None
        with torch.no_grad():
            for i in range(grasp_trial_num):
                pos      = poses[i, 0:3]
                axis     = poses[i, 3:6]
                approach = poses[i, 6:9]
                binormal = poses[i, 9:12]
                width = poses[i, 12] 
                R_old = np.column_stack([-axis, binormal, approach])
            
                if not utils.compute_approach_angle(R_old[:3, 2]):
                    # _, R_old = utils.adjust_pose_z_axis_to_down(R_old)
                    # print('current grasp is down!!!')
                    continue
                pose_excute = np.eye(4)
                pose_excute[:3, :3] = R_old
                pose_excute[:3, 3] = pos
                pose_excutes.append(pose_excute)
                widths.append(width)
                quaternion = R.from_matrix(pose_excute[:3 ,:3]).as_quat()
                record_pose = np.hstack([pose_excute[:3 ,3], quaternion])
                pose = torch.from_numpy(record_pose).float()
                sence_points = utils.TransformPCD2EndLink(global_sv_ply, pose)
                pose_points, _ = utils.gripper_point_width(n_target=200, gripper_width=width)
        
                crop_mask, _ = utils.get_mask_bottom_origin_z_axis(local_points=sence_points, radius=(width/2 + 0.0143), height=0.1, extend=0.008, angle_deg=30)
                # crop_mask = torch.from_numpy(crop_mask).to(device=sence_points.device) 
                crop_extend_points = sence_points[crop_mask]
                crop_extend_points_sample, _ = utils.furthest_point_sampling_nocuda(crop_extend_points, n_samples=720)
                gripper_close_pc, index = utils.fuse_state_torch_v3(crop_extend_points_sample, pose_points)
                if len(gripper_close_pc) < 50:
                    continue
                _, sample_index = utils.furthest_point_sampling_nocuda(gripper_close_pc, n_samples=345)
                close_points_index = index[sample_index]
                uniform_fuse_point, _, _ = utils.pc_normalize_grasp(crop_extend_points_sample)
                uniform_fuse_point = uniform_fuse_point.T.to(dtype=torch.float32)
                uniform_fuse_points.append(uniform_fuse_point)
                close_points_indexs.append(close_points_index)
            if len(uniform_fuse_points) == 0:
                return False, None, None
            batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0) # Nx3 -> 10xNx3
            batch_close_points_indexs = torch.stack(close_points_indexs, dim=0)
            # batch_uniform_fuse_points = batch_uniform_fuse_points.transpose(1, 2) # 10xNx3 -> 10x3xN
            batch_uniform_fuse_points = batch_uniform_fuse_points.to('cuda')
            batch_close_points_indexs = batch_close_points_indexs.to('cuda')
            pred = self.model(batch_uniform_fuse_points, batch_close_points_indexs)
            pred_1 = pred['grasp_cls_pred']
            pred_2 = pred['depth_rotation_cls_pred']
            pred_1 = F.softmax(pred_1, dim=1)
            pred_class = pred_1.data.max(1, keepdim=True)[1]
        state_evaluate = False
        action_idxs = (pred_class == 1).nonzero(as_tuple=True)[0]
        if len(action_idxs) > 0:
        # rank the succeessful actions and select the best action to execute
            best_pre, best_action_idx = torch.max(pred_1[:,1], dim=0)
            if best_pre >= 0.65:
                state_evaluate = True
            grasp_pose = pose_excutes[best_action_idx]
            print(f"\033[36m best_grasp_pre = {best_pre} \033[0m")
            return state_evaluate, grasp_pose, widths[best_action_idx]
        else:

            print('\033[31m No valid grasp pose at that state! \033[0m')
            return False, None, None

    def evalueate_grasp_actions_return_bestpre(self, global_sv_ply, poses):
        grasp_trial_num = len(poses)
        uniform_fuse_points = []
        close_points_indexs = []
        pose_excutes = []
        widths = []
        self.model.eval()
        if grasp_trial_num < 1:
            print('\033[32m No grasp pose be generated at current state! \033[0m')
            return False, None, None, None
        with torch.no_grad():
            for i in range(grasp_trial_num):
                pos      = poses[i, 0:3]
                axis     = poses[i, 3:6]
                approach = poses[i, 6:9]
                binormal = poses[i, 9:12]
                width = poses[i, 12] 
                R_old = np.column_stack([-axis, binormal, approach])
            
                if not utils.compute_approach_angle(R_old[:3, 2]):
                    # _, R_old = utils.adjust_pose_z_axis_to_down(R_old)
                    # print('current grasp is down!!!')
                    continue
                pose_excute = np.eye(4)
                pose_excute[:3, :3] = R_old
                pose_excute[:3, 3] = pos
                pose_excutes.append(pose_excute)
                widths.append(width)
                quaternion = R.from_matrix(pose_excute[:3 ,:3]).as_quat()
                record_pose = np.hstack([pose_excute[:3 ,3], quaternion])
                pose = torch.from_numpy(record_pose).float()
                sence_points = utils.TransformPCD2EndLink(global_sv_ply, pose)
                pose_points, _ = utils.gripper_point_width(n_target=200, gripper_width=width)
        
                crop_mask, _ = utils.get_mask_bottom_origin_z_axis(local_points=sence_points, radius=(width/2 + 0.0143), height=0.1, extend=0.008, angle_deg=30)
                # crop_mask = torch.from_numpy(crop_mask).to(device=sence_points.device) 
                crop_extend_points = sence_points[crop_mask]
                crop_extend_points_sample, _ = utils.furthest_point_sampling_nocuda(crop_extend_points, n_samples=960)
                gripper_close_pc, index = utils.fuse_state_torch_v3(crop_extend_points_sample, pose_points)
                if len(gripper_close_pc) < 50:
                    continue
                _, sample_index = utils.furthest_point_sampling_nocuda(gripper_close_pc, n_samples=345)
                close_points_index = index[sample_index]
                uniform_fuse_point, _, _ = utils.pc_normalize_grasp(crop_extend_points_sample)
                uniform_fuse_point = uniform_fuse_point.T.to(dtype=torch.float32)
                uniform_fuse_points.append(uniform_fuse_point)
                close_points_indexs.append(close_points_index)
            if len(uniform_fuse_points) == 0:
                return False, None, None, None
            batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0) # Nx3 -> 10xNx3
            batch_close_points_indexs = torch.stack(close_points_indexs, dim=0)
            # batch_uniform_fuse_points = batch_uniform_fuse_points.transpose(1, 2) # 10xNx3 -> 10x3xN
            batch_uniform_fuse_points = batch_uniform_fuse_points.to('cuda')
            batch_close_points_indexs = batch_close_points_indexs.to('cuda')
            pred = self.model(batch_uniform_fuse_points, batch_close_points_indexs)
            pred_1 = pred['grasp_cls_pred']
            pred_2 = pred['depth_rotation_cls_pred']
            pred_1 = F.softmax(pred_1, dim=1)
            pred_class = pred_1.data.max(1, keepdim=True)[1]
        state_evaluate = False
        action_idxs = (pred_class == 1).nonzero(as_tuple=True)[0]
        if len(action_idxs) > 0:
        # rank the succeessful actions and select the best action to execute
            best_pre, best_action_idx = torch.max(pred_1[:,1], dim=0)
            if best_pre >= 0.65:
                state_evaluate = True
            grasp_pose = pose_excutes[best_action_idx]
            print(f"\033[36m best_grasp_pre = {best_pre} \033[0m")
            return state_evaluate, grasp_pose, widths[best_action_idx], best_pre
        else:

            print('\033[31m No valid grasp pose at that state! \033[0m')
            return False, None, None, None
    


    # def evalueate_grasp_actions_fast(self, global_sv_ply, poses):
    #     def sync():
    #         if torch.cuda.is_available():
    #             torch.cuda.synchronize()
            

    #     t0 = time.time()
    #     grasp_trial_num = len(poses)
    #     uniform_fuse_points = []
    #     close_points_indexs = []
    #     pose_excutes = []
    #     widths = []
    #     self.model.eval()
    #     if grasp_trial_num < 1:
    #         print('\033[32m No grasp pose be generated at current state! \033[0m')
    #         return False, None, None, None, None, None
    #     t1 = time.time()
    #     print(f'T0:{t1-t0}')
    #     # with torch.no_grad():
    #     with torch.inference_mode():
    #         for i in range(grasp_trial_num):
    #             sync()
    #             T0 = time.perf_counter()
    #             pos      = poses[i, 0:3]
    #             axis     = poses[i, 3:6]
    #             approach = poses[i, 6:9]
    #             binormal = poses[i, 9:12]
    #             width = poses[i, 12] 
    #             R_old = np.column_stack([-axis, binormal, approach])
            
    #             if not utils.compute_approach_angle(R_old[:3, 2]):
    #                 # _, R_old = utils.adjust_pose_z_axis_to_down(R_old)
    #                 # print('current grasp is down!!!')
    #                 continue
    #             sync()
    #             T1 = time.perf_counter()
    #             print(f't1:{T1-T0}')
    #             pose_excute = np.eye(4)
    #             pose_excute[:3, :3] = R_old
    #             pose_excute[:3, 3] = pos
    #             pose_excutes.append(pose_excute)
    #             widths.append(width)
    #             t = torch.as_tensor(pos, device=global_sv_ply.device, dtype=global_sv_ply.dtype)          # [3]
    #             R = torch.as_tensor(R_old, device=global_sv_ply.device, dtype=global_sv_ply.dtype)

    #             # SVD 正交化：把近似旋转投影为合法旋转
    #             U, _, Vh = torch.linalg.svd(R)
    #             R = U @ Vh
    #             # 修正 det = +1（避免反射）
    #             if torch.linalg.det(R) < 0:
    #                 U[:, -1] *= -1
    #                 R = U @ Vh

    #             sence_points = (global_sv_ply - t.unsqueeze(0)) @ R
    #             pose_points = utils.get_gripper_points(width, device=global_sv_ply.device, dtype=global_sv_ply.dtype)  # [200,3]
    #             radius = float(width / 2.0 + 0.0143)
    #             crop_mask, _ = utils.get_mask_bottom_origin_z_axis_torch(
    #                 sence_points, radius=radius, height=0.1, extend=0.008, angle_deg=30.0
    #             )
    #             crop_extend_points = sence_points[crop_mask]
    #             sync()
    #             T2 = time.perf_counter()
    #             print(f't2:{T2-T1}')

    #             crop_extend_points_sample, _ = utils.fps_p3d(crop_extend_points, n_samples=960)
    #             sync()
    #             T3 = time.perf_counter()
    #             print(f't3:{T3-T2}')

    #             # print(f'sample time1:{T1-T0}')
    #             gripper_close_pc, index = utils.fuse_state_torch_v3(crop_extend_points_sample, pose_points)
    #             if len(gripper_close_pc) < 50:
    #                 continue
    #             sync()
    #             T4 = time.perf_counter()
    #             print(f't4:{T4-T3}')

    #             _, sample_index = utils.fps_p3d(gripper_close_pc, n_samples=345)
    #             sync()
    #             T5 = time.perf_counter()
    #             print(f't5:{T5-T4}')

    #             # print(f'sample time1:{T3-T2}')
    #             close_points_index = index[sample_index]
    #             uniform_fuse_point, _, _ = utils.pc_normalize_grasp(crop_extend_points_sample)
    #             uniform_fuse_point = uniform_fuse_point.T.to(dtype=torch.float32)
    #             uniform_fuse_points.append(uniform_fuse_point)
    #             close_points_indexs.append(close_points_index)
    #             sync()
    #             T6 = time.perf_counter()
    #             print(f't6:{T6-T5}')
    #             print(f't7:{T6-T0}')


    #         if len(uniform_fuse_points) == 0:
    #             return False, None, None, None, None, None
    #         t2 = time.time()
    #         print(f"T1:{t2-t1}")
    #         # batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0) # Nx3 -> 10xNx3
    #         # batch_close_points_indexs = torch.stack(close_points_indexs, dim=0)
    #         batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0).contiguous().to(self.device, non_blocking=True)
    #         batch_close_points_indexs = torch.stack(close_points_indexs, dim=0).contiguous().to(self.device, non_blocking=True)
    #         # torch.backends.cudnn.benchmark = False
    #         # batch_uniform_fuse_points = batch_uniform_fuse_points.transpose(1, 2) # 10xNx3 -> 10x3xN
    #         starter = torch.cuda.Event(enable_timing=True)
    #         ender   = torch.cuda.Event(enable_timing=True)
    #         torch.cuda.synchronize()
    #         starter.record()
    #         pred = self.model(batch_uniform_fuse_points, batch_close_points_indexs)
    #         ender.record()
    #         torch.cuda.synchronize()
    #         print(f"\033[35m model infernce time cost is {starter.elapsed_time(ender) / 1000.0} \033[0m")
    #         pred_1 = pred['grasp_cls_pred']
    #         pred_2 = pred['depth_rotation_cls_pred']
    #         flat_idx = torch.argmax(pred_2)     
    #         refin_row = (flat_idx // pred_2.size(1)).item()   
    #         refine_col = (flat_idx %  pred_2.size(1)).item()
    #         refine_val = pred_2[refin_row, refine_col].item()
    #         refine_val_prob = torch.sigmoid(torch.tensor(refine_val))
    #         pred_1 = F.softmax(pred_1, dim=1)
    #         pred_class = pred_1.data.max(1, keepdim=True)[1]
    #         t3 = time.time()
    #         # print(f'T2:{t3-t2}')
    #     refine = False
    #     state_evaluate = False
    #     # action_idxs = (pred_class == 1).nonzero(as_tuple=True)[0]
    #     # if len(action_idxs) > 0:
    #     # rank the succeessful actions and select the best action to execute
    #     best_pre, best_action_idx = torch.max(pred_1[:,1], dim=0)
    #     if best_pre > 0.6:
    #         state_evaluate = True
    #     elif refine_val_prob > 0.6:
    #         print(f"\033[36m Enter refine mode!!! \033[0m")
    #         state_evaluate = True
    #         refine = True
    #         best_action_idx = refin_row

    #     grasp_pose = pose_excutes[best_action_idx]
    #     print(f"\033[36m best_grasp_pre = {best_pre} \033[0m")
    #     print(f"\033[36m best_refine_val = {refine_val_prob} \033[0m")
    #     return state_evaluate, grasp_pose, widths[best_action_idx], best_pre, refine, refine_col
    #     # else:
    #     #     print('\033[31m No valid grasp pose at that state! \033[0m')
    #     #     return False, None, None, None, None, None


    def evalueate_grasp_actions_fast(self, global_sv_ply, poses):
        def sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            

        t0 = time.time()
        poses = np.asarray(poses)
        grasp_trial_num = len(poses)
        self.model.eval()
        if grasp_trial_num < 1:
            print('\033[32m No grasp pose be generated at current state! \033[0m')
            return False, None, None, None, None, None
        t1 = time.time()
        with torch.inference_mode():
            point_device = global_sv_ply.device
            point_dtype = global_sv_ply.dtype
            model_device = torch.device(self.device)
            approach_np = poses[:, 6:9]
            approach_norm = np.linalg.norm(approach_np, axis=1)
            valid_mask = approach_norm > 1e-12
            valid_mask &= (np.abs(approach_np[:, 2]) / np.maximum(approach_norm, 1e-12)) >= math.sin(math.radians(45.0))
            poses = poses[valid_mask]

            if len(poses) == 0:
                return False, None, None, None, None, None

            pos_np = poses[:, 0:3]
            axis_np = poses[:, 3:6]
            approach_np = poses[:, 6:9]
            binormal_np = poses[:, 9:12]
            widths_np = poses[:, 12].astype(np.float32, copy=False)
            width_mm_np = np.rint(widths_np * 1000.0).astype(np.int32, copy=False)
            rotation_np = np.stack([-axis_np, binormal_np, approach_np], axis=2)

            pose_excutes_np = np.tile(np.eye(4, dtype=np.float64), (len(poses), 1, 1))
            pose_excutes_np[:, :3, :3] = rotation_np
            pose_excutes_np[:, :3, 3] = pos_np

            t_batch = torch.as_tensor(pos_np, device=point_device, dtype=point_dtype)
            R_batch = torch.as_tensor(rotation_np, device=point_device, dtype=point_dtype)
            R_batch = orthogonalize_rot_batch(R_batch)

            sence_points_batch = (global_sv_ply.unsqueeze(0) - t_batch[:, None, :]) @ R_batch
            radius_batch = torch.as_tensor(widths_np * 0.5 + 0.0143, device=point_device, dtype=point_dtype)
            crop_mask_batch, _ = get_mask_bottom_origin_z_axis_torch_batch(
                sence_points_batch,
                radius=radius_batch,
                height=0.1,
                extend=0.008,
                angle_deg=30.0,
            )
            t_crop = time.time()

            crop_points_for_fps_list = []
            crop_meta_ids = []
            for pose_i in range(sence_points_batch.shape[0]):
                crop_extend_points = sence_points_batch[pose_i][crop_mask_batch[pose_i]]
                if crop_extend_points.shape[0] < 50:
                    continue

                crop_points_for_fps, _, _ = utils.pre_voxel_downsample_for_fps(
                    crop_extend_points,
                    cap=1050,
                    trigger=1200,
                    voxel_size0=0.002,
                    growth=1.35,
                    max_iter=6,
                )
                if crop_points_for_fps.shape[0] < 50:
                    continue

                crop_points_for_fps_list.append(crop_points_for_fps)
                crop_meta_ids.append(pose_i)

            if len(crop_points_for_fps_list) == 0:
                return False, None, None, None, None, None
            t_voxel = time.time()

            crop_extend_points_sample_batch, _, _ = utils.fps_p3d_batch_from_list(
                crop_points_for_fps_list,
                n_samples=960,
            )
            t_fps1 = time.time()

            obb_param_cache = {}
            for width_mm in np.unique(width_mm_np[crop_meta_ids]):
                center, axes, half_extent = utils.get_gripper_obb_params(
                    width=float(width_mm) / 1000.0,
                    device=point_device,
                    dtype=point_dtype,
                )
                obb_param_cache[int(width_mm)] = (center, axes, half_extent)

            close_points_indexs = []
            gripper_close_points = []
            gripper_index_maps = []
            successful_batch_ids = []
            successful_pose_ids = []
            for batch_i, pose_i in enumerate(crop_meta_ids):
                crop_extend_points_sample = crop_extend_points_sample_batch[batch_i]
                center, axes, half_extent = obb_param_cache[int(width_mm_np[pose_i])]
                gripper_close_pc, index = utils.fuse_state_torch_obb_params(
                    crop_extend_points_sample,
                    center=center,
                    axes=axes,
                    half_extent=half_extent,
                )
                if len(gripper_close_pc) < 50:
                    continue

                gripper_close_points.append(gripper_close_pc)
                gripper_index_maps.append(index)
                successful_batch_ids.append(batch_i)
                successful_pose_ids.append(pose_i)

            if len(successful_batch_ids) == 0:
                return False, None, None, None, None, None
            t_fuse = time.time()

            _, sample_index_batch, _ = utils.fps_p3d_batch_from_list(
                gripper_close_points,
                n_samples=345,
            )

            padded_index_maps, _ = utils.pad_pointcloud_list(
                [index.unsqueeze(1) for index in gripper_index_maps],
                pad_value=0,
            )
            batch_ids = torch.arange(sample_index_batch.shape[0], device=sample_index_batch.device)[:, None]
            batch_close_points_indexs = padded_index_maps[batch_ids, sample_index_batch].squeeze(-1).contiguous()

            selected_batch_ids = torch.as_tensor(
                successful_batch_ids,
                device=crop_extend_points_sample_batch.device,
                dtype=torch.long,
            )
            sampled_crops_batch = crop_extend_points_sample_batch[selected_batch_ids]
            uniform_fuse_points, _, _ = pc_normalize_grasp_batch(sampled_crops_batch)
            batch_uniform_fuse_points = uniform_fuse_points.transpose(1, 2).contiguous()
            if batch_uniform_fuse_points.device != model_device or batch_uniform_fuse_points.dtype != torch.float32:
                batch_uniform_fuse_points = batch_uniform_fuse_points.to(
                    model_device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            if batch_close_points_indexs.device != model_device:
                batch_close_points_indexs = batch_close_points_indexs.to(
                    model_device,
                    non_blocking=True,
                )
            t2 = time.time()
            print(f"Tpre:{t2-t1}")
            print(
                "Tpre_detail:"
                f" crop={t_crop-t1:.4f}"
                f" voxel={t_voxel-t_crop:.4f}"
                f" fps1={t_fps1-t_voxel:.4f}"
                f" fuse={t_fuse-t_fps1:.4f}"
                f" fps2_norm={t2-t_fuse:.4f}"
                f" keep={len(successful_batch_ids)}/{len(poses)}"
            )

            if torch.cuda.is_available():
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                sync()
                starter.record()
                pred = self.model(batch_uniform_fuse_points, batch_close_points_indexs)
                ender.record()
                sync()
                model_time = starter.elapsed_time(ender) / 1000.0
            else:
                infer_t0 = time.time()
                pred = self.model(batch_uniform_fuse_points, batch_close_points_indexs)
                model_time = time.time() - infer_t0
            print(f"\033[35m model infernce time cost is {model_time} \033[0m")
            pred_1 = pred['grasp_cls_pred']
            pred_2 = pred['depth_rotation_cls_pred']
            flat_idx = torch.argmax(pred_2)     
            refin_row = (flat_idx // pred_2.size(1)).item()   
            refine_col = (flat_idx %  pred_2.size(1)).item()
            refine_val = pred_2[refin_row, refine_col].item()
            refine_val_prob = torch.sigmoid(torch.tensor(refine_val))
            pred_1 = F.softmax(pred_1, dim=1)
            t3 = time.time()
            # print(f'T2:{t3-t2}')
        refine = False
        state_evaluate = False
        # action_idxs = (pred_class == 1).nonzero(as_tuple=True)[0]
        # if len(action_idxs) > 0:
        # rank the succeessful actions and select the best action to execute
        best_pre, best_action_idx = torch.max(pred_1[:,1], dim=0)
        best_action_idx = int(best_action_idx.item())
        # if best_pre > 0.6:
        #     state_evaluate = True
        if refine_val_prob > 0.6:
            print(f"\033[36m Enter refine mode!!! \033[0m")
            state_evaluate = True
            refine = True
            best_action_idx = refin_row

        pose_excutes = pose_excutes_np[np.asarray(successful_pose_ids, dtype=np.int64)]
        widths = widths_np[np.asarray(successful_pose_ids, dtype=np.int64)]
        grasp_pose = pose_excutes[best_action_idx]
        print(f"\033[36m best_grasp_pre = {best_pre} \033[0m")
        print(f"\033[36m best_refine_val = {refine_val_prob} \033[0m")
        return state_evaluate, grasp_pose, widths[best_action_idx], best_pre, refine, refine_col
        # else:
        #     print('\033[31m No valid grasp pose at that state! \033[0m')
        #     return False, None, None, None, None, None

    def evalueate_grasp_actions_fast_v2(self, global_sv_ply, poses):
        t0 = time.time()
        grasp_trial_num = len(poses)
        uniform_fuse_points = []
        close_points_indexs = []
        pose_excutes = []
        widths = []
        self.model.eval()
        if grasp_trial_num < 1:
            print('\033[32m No grasp pose be generated at current state! \033[0m')
            return False, None, None, None, None, None
        t1 = time.time()
        # print(f'T0:{t1-t0}')
        # with torch.no_grad():
        with torch.inference_mode():
            for i in range(grasp_trial_num):
                T0 = time.time()
                pos      = poses[i, 0:3]
                axis     = poses[i, 3:6]
                approach = poses[i, 6:9]
                binormal = poses[i, 9:12]
                width = poses[i, 12] 
                R_old = np.column_stack([-axis, binormal, approach])
            
                if not utils.compute_approach_angle(R_old[:3, 2]):
                    # _, R_old = utils.adjust_pose_z_axis_to_down(R_old)
                    # print('current grasp is down!!!')
                    continue
                T1 = time.time()
                # print(f't1:{T1-T0}')
                pose_excute = np.eye(4)
                pose_excute[:3, :3] = R_old
                pose_excute[:3, 3] = pos
                pose_excutes.append(pose_excute)
                widths.append(width)
                t = torch.as_tensor(pos, device=global_sv_ply.device, dtype=global_sv_ply.dtype)          # [3]
                R = torch.as_tensor(R_old, device=global_sv_ply.device, dtype=global_sv_ply.dtype)

                # SVD 正交化：把近似旋转投影为合法旋转
                U, _, Vh = torch.linalg.svd(R)
                R = U @ Vh
                # 修正 det = +1（避免反射）
                if torch.linalg.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vh

                sence_points = (global_sv_ply - t.unsqueeze(0)) @ R
                pose_points = utils.get_gripper_points(width, device=global_sv_ply.device, dtype=global_sv_ply.dtype)  # [200,3]
                radius = float(width / 2.0 + 0.0143)
                crop_mask, _ = utils.get_mask_bottom_origin_z_axis_torch(
                    sence_points, radius=radius, height=0.1, extend=0.008, angle_deg=30.0
                )
                crop_extend_points = sence_points[crop_mask]
                if crop_extend_points is None or crop_extend_points.shape[0] == 0:
                    continue
                T2 = time.time()
                # print(f't2:{T2-T1}')

                crop_extend_points_sample, _ = utils.furthest_point_sampling_nocuda(crop_extend_points, n_samples=960)

                T3 = time.time()
                # print(f't3:{T3-T2}')

                # print(f'sample time1:{T1-T0}')
                gripper_close_pc, index = utils.fuse_state_torch_v3(crop_extend_points_sample, pose_points)
                if len(gripper_close_pc) < 50:
                    continue
                T4 = time.time()
                # print(f't4:{T4-T3}')

                _, sample_index = utils.furthest_point_sampling_nocuda(gripper_close_pc, n_samples=345)
                T5 = time.time()
                # print(f't5:{T5-T4}')

                # print(f'sample time1:{T3-T2}')
                close_points_index = index[sample_index]
                uniform_fuse_point, _, _ = utils.pc_normalize_grasp(crop_extend_points_sample)
                uniform_fuse_point = uniform_fuse_point.T.to(dtype=torch.float32)
                uniform_fuse_points.append(uniform_fuse_point)
                close_points_indexs.append(close_points_index)
                T6 = time.time()
                # print(f't6:{T6-T5}')
                # print(f't7:{T6-T0}')


            if len(uniform_fuse_points) == 0:
                return False, None, None, None, None, None
            t2 = time.time()
            # print(f"T1:{t2-t1}")
            # batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0) # Nx3 -> 10xNx3
            # batch_close_points_indexs = torch.stack(close_points_indexs, dim=0)
            batch_uniform_fuse_points = torch.stack(uniform_fuse_points, dim=0).contiguous().to(self.device, non_blocking=True)
            batch_close_points_indexs = torch.stack(close_points_indexs, dim=0).contiguous().to(self.device, non_blocking=True)
            # torch.backends.cudnn.benchmark = False
            # batch_uniform_fuse_points = batch_uniform_fuse_points.transpose(1, 2) # 10xNx3 -> 10x3xN
            # starter = torch.cuda.Event(enable_timing=True)
            # ender   = torch.cuda.Event(enable_timing=True)
            # torch.cuda.synchronize()
            # starter.record()
            pred = self.model(batch_uniform_fuse_points, batch_close_points_indexs)
            # ender.record()
            # torch.cuda.synchronize()
            # print(f"\033[35m model infernce time cost is {starter.elapsed_time(ender) / 1000.0} \033[0m")
            pred_1 = pred['grasp_cls_pred']
            pred_2 = pred['depth_rotation_cls_pred']
            flat_idx = torch.argmax(pred_2)     
            refin_row = (flat_idx // pred_2.size(1)).item()   
            refine_col = (flat_idx %  pred_2.size(1)).item()
            refine_val = pred_2[refin_row, refine_col].item()
            refine_val_prob = torch.sigmoid(torch.tensor(refine_val))
            pred_1 = F.softmax(pred_1, dim=1)
            pred_class = pred_1.data.max(1, keepdim=True)[1]
            t3 = time.time()
            # print(f'T2:{t3-t2}')
        refine = False
        state_evaluate = False
        # action_idxs = (pred_class == 1).nonzero(as_tuple=True)[0]
        # if len(action_idxs) > 0:
        # rank the succeessful actions and select the best action to execute
        best_pre, best_action_idx = torch.max(pred_1[:,1], dim=0)
        if best_pre > 0.8:
            state_evaluate = True
        elif refine_val_prob > 0.8:
            print(f"\033[36m Enter refine mode!!! \033[0m")
            state_evaluate = True
            refine = True
            best_action_idx = refin_row

        grasp_pose = pose_excutes[best_action_idx]
        print(f"\033[36m best_grasp_pre = {best_pre} \033[0m")
        print(f"\033[36m best_refine_val = {refine_val_prob} \033[0m")
        return state_evaluate, grasp_pose, widths[best_action_idx], best_pre, refine, refine_col
        # else:
        #     print('\033[31m No valid grasp pose at that state! \033[0m')
        #     return False, None, None, None, None, None




    
