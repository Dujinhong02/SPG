from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_encoder import Encoder_PointCloud,Encoder_PointCloud_Type
import numpy as np

class Space_GraspFusion(nn.Module):
    def __init__(self, device,k=2):
        super().__init__()
        
        self.device = device

        self.pointnet = Encoder_PointCloud_Type() # 超参4

        self.fc1 = nn.Linear(1024, 512) # 超参5
        self.drop1 = nn.Dropout(0.4) # 新增
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

    def encode_SpacePointCloud(self, x):
        spacepointcloud = self.pointnet(x.to(self.device))
        return spacepointcloud
    

    def forward(self, pointcloud):

        space_feat = self.encode_SpacePointCloud(pointcloud)
        space_feat = space_feat.squeeze(-1)
        # B,N,C = space_feat.shape
        x = F.relu(self.bn1(self.fc1(space_feat)))
        x = self.drop1(x) # 新增
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        # return F.log_softmax(x, dim=-1)
        return x
    
# class Space_GraspFusion(nn.Module):
#     def __init__(self, device,k=2):
#         super().__init__()
        
#         self.device = device

#         self.pointnet = Encoder_PointCloud() # 超参4

#         self.fc1 = nn.Linear(1024, 512) # 超参5
#         self.drop1 = nn.Dropout(0.4) # 新增
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, k)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(128)

#         self.relu = nn.ReLU()

#     def encode_SpacePointCloud(self, x):
#         spacepointcloud = self.pointnet(x.to(self.device))
#         return spacepointcloud
    

#     def forward(self, pointcloud):

#         space_feat = self.encode_SpacePointCloud(pointcloud)
#         space_feat = space_feat.squeeze(-1)
#         # B,N,C = space_feat.shape
#         x = F.relu(self.bn1(self.fc1(space_feat)))
#         x = self.drop1(x) # 新增
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = self.fc3(x)
#         # return F.log_softmax(x, dim=-1)
#         return x

# class Space_GraspFusion_FiLM(nn.Module):
#     def __init__(self, device,k=2):
#         super().__init__()
        
#         self.device = device

#         self.pointnet = Encoder_PointCloud_FiLM() # 超参4
#         # self.film = FiLM(512)
#         self.fc1 = nn.Linear(1024, 512) # 超参5
#         self.drop1 = nn.Dropout(0.4) # 新增
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, k)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(128)

#         self.relu = nn.ReLU()

#     def encode_SpacePointCloud(self, x, y):
#         gripper_feature, scene_feature = self.pointnet(x.to(self.device), y.to(self.device))
#         return gripper_feature, scene_feature
    

#     def forward(self, gripper_pc, scene_pc):

#         gripper_feature, scene_feature = self.encode_SpacePointCloud(gripper_pc, scene_pc)
#         gripper_feature = gripper_feature.squeeze(-1)
#         scene_feature = scene_feature.squeeze(-1)
#         # scene_feature_film = self.film(gripper_feature, scene_feature)
#         space_feat = torch.cat([gripper_feature, scene_feature],dim=1)
#         # B,N,C = space_feat.shape
#         x = F.relu(self.bn1(self.fc1(space_feat)))
#         x = self.drop1(x) # 新增
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = self.fc3(x)
#         # return F.log_softmax(x, dim=-1)
#         return x

# class FiLM(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.gamma_gen = nn.Linear(in_channels, in_channels)
#         self.beta_gen = nn.Linear(in_channels, in_channels)

#     def forward(self, cond, feat):
#         gamma = self.gamma_gen(cond)
#         beta = self.beta_gen(cond)
#         return gamma * feat + beta  

    
class S_A_Fusion(nn.Module):
    """
    global pc -> gloabl feature 1XW
    object pc -> object feature 1XW
    push_action -> push feature NXW
    fuse gloabl feature and object feature into push feature -> push feature Nxd
    """
    def __init__(self, push_dim, width, device):
        super().__init__()

        self.device = device
        self.pointnet_global = Encoder_PointCloud_SAC(width * 2)
        self.pointnet_obj = Encoder_PointCloud_SAC(width)
        # self.obj_pc_encoder = Encoder_PointCloud()
        self.push_action_mlp = nn.Sequential(
                                nn.Linear(push_dim, 128),
                                nn.ReLU(),
                                nn.Linear(128, width),
                                nn.ReLU(),
                                nn.Linear(width, width)
                                )
    
    def encoder_pc_global(self, x):
        # x = torch.from_numpy(x).float()
        # x = x.unsqueeze(0) 
        # x = x.transpose(1,2)
        global_pc_feature = self.pointnet_global(x.to(self.device))
        return global_pc_feature
    
    def encoder_pc_obj(self, x):
        # x = torch.from_numpy(np.array(x.points)).float()
        # x = x.unsqueeze(0) 
        # x = x.transpose(1,2)
        obj_pc_feature = self.pointnet_obj(x.to(self.device))
        return obj_pc_feature

    def forward(self,global_pc, obj_pc, push_actions):
        """
            global -> 1xNx3
            push_actions -> Nx7
        """
        N = push_actions.shape[0]
        global_feature = self.encoder_pc_global(global_pc) # 1xC*2
        global_feature = global_feature.repeat(1, 1, N)
        global_feature = global_feature.transpose(1, 2)

        obj_feature = self.encoder_pc_obj(obj_pc) # 1xC
        obj_feature = obj_feature.repeat(1, 1, N)
        obj_feature = obj_feature.transpose(1, 2)
        # push_actions = torch.from_numpy(push_actions).float()
        push_actions = push_actions.unsqueeze(0) # Nx7 -> 1xNx7
        push_actions = push_actions.to(self.device) 
        push_actions_feature = self.push_action_mlp(push_actions) # 1xNxC
        # push_actions_feature = push_actions_feature.unsqueeze(0)
        fused_feature = torch.cat([global_feature, obj_feature, push_actions_feature], dim=2)
        return fused_feature  # 1 x N x 1024


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim *2, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim * 2)
        self.linear5 = nn.Linear(hidden_dim *2, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, sa):
        
        x1 = F.relu(self.linear1(sa))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(sa))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2
    
class Policy(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        logits = self.linear3(x).squeeze()
        return logits

