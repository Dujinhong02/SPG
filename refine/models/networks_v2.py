from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from refine.models.pointnet2_encoder import Encoder_PointCloud2, GraspableNet, Crop_Group_Net, RefineNet
import numpy as np

class Space_GraspFusion(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.pointnet = Encoder_PointCloud2() 
        self.crop_group = Crop_Group_Net()
        self.graspable_net = GraspableNet()
        self.refine_net = RefineNet()
    def encode_SpacePointCloud(self, x):
        spacepointcloud = self.pointnet(x.to(self.device))
        return spacepointcloud
    
    def forward(self, pointcloud, labels):
        # points_index = labels['points_index']          # (B,2,K)  LongTensor
        # points_index1 = points_index[:, 0, :]          # (B,K)
        # points_index2 = points_index[:, 1, :]          # (B,K)
        # points_index = torch.stack([points_index1, points_index2], dim=1)  # (B,2,K)
        # pointcloud = pointcloud.to('cuda')
        space_feat = self.encode_SpacePointCloud(pointcloud) # [B, N, 128]
        crop_feature = self.crop_group(space_feat, pointcloud, labels) # (B,256,2) 
        # crop_feature1 = crop_feature[..., 0]   # (B,256,1)
        # crop_feature2 = crop_feature[..., 1] 
        # x = self.graspable_net(crop_feature1)
        # y, z, w = self.refine_net(crop_feature)
        # crop_feature = crop_feature.squeeze(-1)
        x = self.graspable_net(crop_feature)
        z = self.refine_net(crop_feature)

        preds = {'grasp_cls_pred':x, 'depth_rotation_cls_pred':z}
        # preds = {'grasp_cls_pred':x, 'refine_cls_pred':y}
        # preds = {'refine_cls_pred':y, 'depth_cls_correct':z}

        return preds
    
