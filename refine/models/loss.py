""" Loss functions for training.
    Reference: chenxi-wang
    Author: Lijingze-Xiao
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
GRASP_MAX_WIDTH = 0.15
def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Author: Charles R. Qi
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss

def compute_grasp_loss(preds, labels, pos_weight_value):

    # 1. graspable cls loss
    criterion_grasp_class = nn.CrossEntropyLoss(reduction='mean')
    grasp_cls_labels = labels['class_labels'][:, 0].long()
    grasp_cls_preds = preds['grasp_cls_pred']
    grasp_class_loss = criterion_grasp_class(grasp_cls_preds, grasp_cls_labels)
    # mask_index = labels['points_index']
    # 2. refineable cls loss
    refine_cls_labels = labels['class_labels'][:, 1].long()
    refine_cls_pred = preds['refine_cls_pred'] 
    refine_mask = (grasp_cls_labels == 0)
    w0 = 1.0
    w1 = float(pos_weight_value)  # 正类权重
    class_weight = torch.tensor([w0, w1], device=refine_cls_pred.device, dtype=torch.float32)

    criterion_refine_class = nn.CrossEntropyLoss(weight=class_weight, reduction='mean')
    if refine_mask.any():
        refine_class_loss = criterion_refine_class(
            refine_cls_pred[refine_mask],
            refine_cls_labels[refine_mask]
        )
    else:
        refine_class_loss = refine_cls_pred.new_zeros(())
    # only successful refine sample can be propagate back to depth and rotation net
    # 3. depth cls loss
    # depth_cls_labels = labels['class_labels'][:, 2].long()
    # depth_cls_labels = depth_cls_labels - 1 # set depth bin to the range of 0-2
    # depth_cls_pred = preds['depth_cls_pred']
    # depth_mask = (grasp_cls_labels == 0) & (refine_cls_labels == 1)
    # criterion_depth_class = nn.CrossEntropyLoss(reduction='mean')
    # depth_class_loss = criterion_depth_class(depth_cls_pred[depth_mask], depth_cls_labels[depth_mask])
    # 4. rotation cls loss
    # rotation_cls_labels = labels['class_labels'][:, 3].long()
    # rotation_cls_labels = rotation_cls_labels + 3 # set rotation bin to the range of 0-6
    # rotation_cls_pred = preds['rotation_cls_pred']
    # rotation_mask = depth_mask
    # criterion_rotation_class = nn.CrossEntropyLoss(reduction='mean')
    # rotation_class_loss = criterion_rotation_class(rotation_cls_pred[rotation_mask], rotation_cls_labels[rotation_mask])
    # 5. sum the loss with weight
    w_refine, w_depth, w_rot = 1.0, 1.0, 1.0
    # loss = grasp_class_loss + w_refine * refine_class_loss + w_depth * depth_class_loss + w_rot * rotation_class_loss
    loss = grasp_class_loss + w_refine * refine_class_loss 


    # record accurate 
    grasp_pre_class = grasp_cls_preds.argmax(dim=1)
    grasp_cls_correct = (grasp_pre_class == grasp_cls_labels).long().cpu().sum().item()

    refine_pre_class = refine_cls_pred.argmax(dim=1)
    refine_cls_correct = (refine_pre_class[refine_mask] == refine_cls_labels[refine_mask]).long().cpu().sum().item()
    # neg sample recall 
    neg_recall_mask = (refine_cls_labels == 0)
    neg_recall_correct = (refine_pre_class[neg_recall_mask] == refine_cls_labels[neg_recall_mask]).long().cpu().sum().item()
    # depth_pre_class = depth_cls_pred.argmax(dim=1)
    # depth_cls_correct = (depth_pre_class[depth_mask] == depth_cls_labels[depth_mask]).long().cpu().sum().item()

    # rotation_pre_class = rotation_cls_pred.argmax(dim=1)
    # rotation_cls_correct = (rotation_pre_class[rotation_mask] == rotation_cls_labels[rotation_mask]).long().cpu().sum().item()

    # real_refine_correct = ((depth_pre_class[depth_mask] == depth_cls_labels[depth_mask]) == (rotation_pre_class[rotation_mask] == rotation_cls_labels[rotation_mask])).long().cpu().sum().item()

    # record_info = {'grasp_cls_correct':grasp_cls_correct, 'refine_cls_correct':refine_cls_correct, 'refine_mask_sum':refine_mask.sum().item(), 'depth_cls_correct':depth_cls_correct, 'rotation_cls_correct':rotation_cls_correct, 'depth_mask_sum':depth_mask.sum().item(), 'real_refine_correct':real_refine_correct}
    record_info = {'grasp_cls_correct':grasp_cls_correct, 'refine_cls_correct':refine_cls_correct, 'refine_mask_sum':refine_mask.sum().item(), 'neg_recall_correct':neg_recall_correct, 'neg_recall_mask_sum':neg_recall_mask.sum().item()}
    
    return loss, record_info