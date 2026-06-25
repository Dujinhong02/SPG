import torch
import torch.nn as nn
import torch.nn.functional as F

from refine.models import pytorch_utils as pt_utils

from refine.models.pointnet2_utils import (
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)
class Encoder_PointCloud(nn.Module):
    def __init__(self, normal_channel=False):
        super(Encoder_PointCloud, self).__init__()
        self.normal_channel = normal_channel
        additional_channel = 3 if normal_channel else 0

        self.sa1 = PointNetSetAbstraction(
            npoint=128,
            radius=0.2,
            nsample=32,
            in_channel=6,
            mlp=[64, 64, 128],
            group_all=False,
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 256, 256],
            group_all=False,
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None,
            in_channel=256 + 3, 
            mlp=[256, 512, 1024], 
            group_all=True)

    def forward(self, xyz):

        if self.normal_channel:
            l0_xyz = xyz[:, :3, :]           
            l0_points = xyz                  
        else:
            l0_xyz = xyz[:, :3, :]          
            l0_points = l0_xyz            # [B, C, N]


        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return l3_points 

class Encoder_PointCloud_Type(nn.Module):
    def __init__(self, add_channel=True):
        super(Encoder_PointCloud_Type, self).__init__()
        # self.normal_channel = normal_channel
        additional_channel = 2 if add_channel else 0

        self.sa1 = PointNetSetAbstraction(
            npoint=128,
            radius=0.2,
            nsample=32,
            in_channel=6 + additional_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 256, 256],
            group_all=False,
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None,
            in_channel=256 + 3, 
            mlp=[256, 512, 1024], 
            group_all=True)

    def forward(self, xyz):
        # xyz-Bx4xN        
        l0_xyz = xyz[:, :3, :]  # [B, C, N]
        l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return l3_points 
    

class Encoder_PointCloud_Type_pose_sence(nn.Module):
    def __init__(self, width=1024,  additional_channel = 0):
        super(Encoder_PointCloud_Type_pose_sence, self).__init__()
        # self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(
            npoint=256,
            radius=0.25,
            nsample=48,
            in_channel=6 + additional_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=64,
            radius=0.5,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 256, 256],
            group_all=False,
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None,
            in_channel=256 + 3, 
            mlp=[256, 512, width], 
            group_all=True)

    def forward(self, xyz):
        # xyz-Bx4xN        
        l0_xyz = xyz[:, :3, :]  # [B, C, N]
        l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return l3_points 

class Encoder_PointCloud_Type_pose_obj(nn.Module):
    def __init__(self, width=1024,  additional_channel = 0):
        super(Encoder_PointCloud_Type_pose_obj, self).__init__()
        # self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(
            npoint=64,
            radius=0.3,
            nsample=32,
            in_channel=6 + additional_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=32,
            radius=0.6,
            nsample=48,
            in_channel=128 + 3,
            mlp=[128, 256, 256],
            group_all=False,
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None,
            in_channel=256 + 3, 
            mlp=[256, 512, width], 
            group_all=True)

    def forward(self, xyz):
        # xyz-Bx4xN        
        l0_xyz = xyz[:, :3, :]  # [B, C, N]
        l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return l3_points 

    # 1.sa3加入导致训练变差 可能的原因：（1）数据集不够（2）超参数调整不合理？
    # 2.sa两层模型训练较为稳定，但成功率不高 80%左右 
    # 可能原因：（1）学习能力不够，网络结构过于简单；
    #        （2）超参数设置不合理：
    #                           1.学习率，decay；
    #                           2.sa网络超参数
    #        （3）数据集？
    #        （4）损失函数？

class Encoder_PointCloud2(nn.Module):
    def __init__(self, addition=False):
        super(Encoder_PointCloud2, self).__init__()
        self.normal_channel = addition
        additional_channel = 2 if addition else 0

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=6 + additional_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + 6 + additional_channel, mlp=[128, 128, 128]
        )

    def forward(self, xyz):
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(
            l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points
        )  # [B, 128, N]

        point_features = l0_points.permute(2, 0, 1)  # [B, N, 128]
        return point_features
    
class Encoder_PointCloud_Push_Type(nn.Module):

    def __init__(self, additional_channel=3):
        super(Encoder_PointCloud_Push_Type, self).__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=6 + additional_channel,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True,
        )
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + 6 + additional_channel, mlp=[128, 128, 128]
        )
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 1, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(
            l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points
        )
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        output = self.drop1(feat)
        output = self.conv2(output)
        output = torch.sigmoid(output)
        output = output.permute(0, 2, 1)
        return output

class Encoder_PointCloud_Push_Type_Loss(nn.Module):

    def __init__(self):

        super(Encoder_PointCloud_Push_Type_Loss, self).__init__()

    def forward(self, pred, target):
        """
        Args:
        - pred : (B, N, 1)  ->  prediction output of model.
        - target : (B, 1)   ->  target value.

        Loss Function :
        - binary_cross_entropy

        Output:
        - loss
        """
        # get shape parameter.
        B, N, _ = pred.shape
        # shape of 'pred_selected' is (B,), indicating the probability of chosen point.
        pred_selected = pred[torch.arange(B), 0, 0]
        # print("prediction_probability is ", pred_selected)
        # change the shape of target from (B, 1) to (B,)
        target = target.view(-1)
        # use binary cross entropy to calculate loss.
        loss = F.binary_cross_entropy(pred_selected, target)

        return loss

class Crop_Group_Net(nn.Module):
    def __init__(self, seed_feature_dim=128+3):
        super(Crop_Group_Net, self).__init__()
        self.in_dim = seed_feature_dim
        mlps = [self.in_dim, 64, 128, 256]
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)
    
    def forward(self, feature, pointcloud, index):
        """ 
        Forward pass.
        feature NxBXC
        pointcloud BXCXN
        """
        B, N = index.shape 
        _, _, C = feature.shape
        feature = feature.permute(1, 0, 2)
        # index = index.to('cuda')
        index_point = index.unsqueeze(-1).expand(-1, -1, 3)  # (B, M, 5)
        index_feature = index.unsqueeze(-1).expand(-1, -1, C)
        pointcloud = pointcloud.permute(0, 2, 1)
        pointcloud = pointcloud[:, :, :3] 
        crop_points = torch.gather(pointcloud, dim=1, index=index_point)  # (B, N, 3)
        crop_feature = torch.gather(feature, dim=1, index=index_feature)  #BxNxC
        crop_feature = torch.cat([crop_feature, crop_points],dim=2)
        crop_feature = crop_feature.permute(0, 2, 1)
        vp_features = self.mlps(
            crop_feature
        ) # vp_features：B C N
        vp_features = F.max_pool1d(
            vp_features, kernel_size=N
        ) 
        # vp_features = F.avg_pool1d(vp_features, kernel_size=N)
        vp_features = vp_features.view(B, -1, 1)
        return vp_features

# class Crop_Group_Net(nn.Module):
#     def __init__(self, seed_feature_dim=128+3):
#         super().__init__()
#         self.in_dim = seed_feature_dim
#         mlps = [self.in_dim, 64, 128, 256]
#         self.mlps = pt_utils.SharedMLP(mlps, bn=True)

#     def forward(self, feature, pointcloud, index):
#         """
#         feature:    (B,P,C) 或 (P,B,C)（会自动纠正）
#         pointcloud: (B,P,3/...) 或 (B,Cpc,P)
#         index:      (B,K) 或 (B,2,K) 或 list([B,K],[B,K])
#         return:     (B,256,1,D) 其中 D=bin 数（1或2）
#         """
#         device = feature.device

#         # ---- 0) 兼容 feature 可能是 (P,B,C) 的旧写法 ----
#         if feature.dim() != 3:
#             raise ValueError(f"feature dim should be 3, got {feature.dim()}")
#         if feature.shape[0] != pointcloud.shape[0] and feature.shape[1] == pointcloud.shape[0]:
#             feature = feature.permute(1, 0, 2)  # -> (B,P,C)

#         B, P, C = feature.shape

#         # ---- 1) pointcloud 统一成 (B,P,3) ----
#         if pointcloud.dim() != 3:
#             raise ValueError(f"pointcloud dim should be 3, got {pointcloud.dim()}")

#         if pointcloud.shape[1] == P and pointcloud.shape[2] >= 3:
#             pc = pointcloud[:, :, :3].to(device)                 # (B,P,3)
#         elif pointcloud.shape[2] == P and pointcloud.shape[1] >= 3:
#             pc = pointcloud.permute(0, 2, 1)[:, :, :3].to(device) # (B,P,3)
#         else:
#             raise ValueError(f"pointcloud/feature 点数不匹配: feature={feature.shape}, pointcloud={pointcloud.shape}")

#         # ---- 2) index 统一成 (B,D,K) ----
#         if isinstance(index, list):
#             index = torch.stack(index, dim=1)  # (B,2,K)

#         index = index.to(device)
#         if index.dtype != torch.long:
#             index = index.long()

#         if index.dim() == 2:
#             index = index.unsqueeze(1)  # (B,1,K)
#         elif index.dim() == 3:
#             # 支持 (B,2,K)；如果你是 (B,K,2) 也可自动纠正
#             if index.shape[2] == 2 and index.shape[1] != 2:
#                 index = index.permute(0, 2, 1)  # -> (B,2,K)
#         else:
#             raise ValueError(f"index dim not supported: {index.dim()}")

#         B, D, K = index.shape  # D=bin数

#         # ---- 3) 不用 expand：flatten 一次 gather ----
#         idx_flat = index.reshape(B, D * K)  # (B, D*K)

#         idx3 = idx_flat.unsqueeze(-1).expand(-1, -1, 3)  # (B,D*K,3)
#         idxC = idx_flat.unsqueeze(-1).expand(-1, -1, C)  # (B,D*K,C)

#         crop_points  = torch.gather(pc,      dim=1, index=idx3).view(B, D, K, 3)  # (B,D,K,3)
#         crop_feature = torch.gather(feature, dim=1, index=idxC).view(B, D, K, C)  # (B,D,K,C)

#         # 拼 xyz： (B,D,K,C+3)
#         crop = torch.cat([crop_feature, crop_points], dim=-1)

#         # SharedMLP 期望 (B,Cin,N)；这里把 (B,D) 合并
#         crop = crop.permute(0, 1, 3, 2).reshape(B * D, C + 3, K)  # (B*D,C+3,K)

#         vp = self.mlps(crop)  # (B*D,256,K)

#         # ---- 4) 每个 bin 各自对 K 点做 max_pool1d ----
#         vp = F.max_pool1d(vp, kernel_size=K)  # (B*D,256,1)

#         # 还原回 (B,256,1,D)
#         vp = vp.view(B, D, 256, 1).permute(0, 2, 3, 1)  # (B,256,1,D)

#         return vp


class GraspableNet(nn.Module):
    """
    class[, ] score[]
    """
    def __init__(self, class_num=2):
        super().__init__()
        self.class_num = class_num

        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, self.class_num, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, L = vp_features.size()
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        if L == 1:
            vp_features = vp_features.squeeze(-1)  

        return vp_features

class RefineNet(nn.Module):
    def __init__(self, refineable=2, depth_rotation_bin=3):
        # Output:
        # tolerance (num_angle)
        super().__init__()
        # num = refineable + depth_bin + rotation_bin
        num = depth_rotation_bin

        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, num, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features):

        B, _, L = vp_features.size()
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        if L == 1:
            vp_features = vp_features.squeeze(-1) 
        # refine_cls = vp_features[:, :2]
        depth_rotation_cls = vp_features[:, 0:]
        # rotation_cls = vp_features[:, 5:]
        # return refine_cls, depth_cls, rotation_cls
        return depth_rotation_cls

    