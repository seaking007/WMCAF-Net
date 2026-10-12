import torch

import numpy as np
import skimage
import medpy.metric
from medpy.metric import hd95
from medpy import  metric



def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    return SE


def get_specificity(SR, GT, threshold=0.5):
    SP = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    return SP


def get_precision(SR, GT, threshold=0.5):
    PC = 0
    SR = SR > threshold
    GT = GT== torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
    return PC





# def iou_score(output, target):
#     smooth = 1e-5
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#
#     intersection = (output_ & target_).sum()
#     union = (output_ | target_).sum()
#     iou = (intersection + smooth) / (union + smooth)
#     dice = (2 * iou) / (iou+1)
#     output_ = torch.tensor(output_)
#     target_ = torch.tensor(target_)
#     SE = get_sensitivity(output_, target_, threshold=0.5)
#     PC = get_precision(output_, target_, threshold=0.5)
#     SP = get_specificity(output_, target_, threshold=0.5)
#     ACC = get_accuracy(output_, target_, threshold=0.5)
#     F1 = 2 * SE * PC/(SE + PC + 1e-6)
#
#
#     return iou, dice, SE, PC, F1, SP, ACC



def get_hd95(SR, GT, threshold=0.5):
    """
    计算 HD95（Hausdorff Distance 95%）指标
    :param SR: 预测分割图 (Tensor or numpy array)，形状 [B, C, H, W] 或 [B, C, D, H, W]
    :param GT: 真实标签 (Tensor or numpy array)
    :param threshold: 二值化阈值，默认 0.5
    :return: 平均 HD95 值
    """
    # 如果是 Tensor，转换为 numpy
    if torch.is_tensor(SR):
        SR = SR.sigmoid().cpu().numpy()
    if torch.is_tensor(GT):
        GT = GT.cpu().numpy()

    SR_bin = SR > threshold
    GT_bin = GT > threshold

    hd_list = []
    for seg, gt in zip(SR_bin, GT_bin):  # 遍历 batch
        # 去除单通道维度 (C=1)
        if seg.shape[0] == 1:
            seg = seg[0]
            gt = gt[0]

        try:
            hd_val = hd95(seg, gt)
        except:
            hd_val = np.inf  # 如果无法计算，返回无穷大
        hd_list.append(hd_val)

    HD95 = np.mean(hd_list)
    return HD95



def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_bin = output > 0.5
    target_bin = target > 0.5

    intersection = (output_bin & target_bin).sum()
    union = (output_bin | target_bin).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)

    # 其它指标（tensor 版）
    output_t = torch.tensor(output_bin)
    target_t = torch.tensor(target_bin)
    SE  = get_sensitivity(output_t, target_t, threshold=0.5)
    PC  = get_precision(output_t, target_t, threshold=0.5)
    SP  = get_specificity(output_t, target_t, threshold=0.5)
    ACC = get_accuracy(output_t, target_t, threshold=0.5)
    F1  = 2 * SE * PC / (SE + PC + 1e-6)
    HD95 = get_hd95(output_t, target_t, threshold=0.5)





    # 计算 HD95（GT, pred）


    return iou, dice, SE, PC, F1, SP, ACC,HD95


def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)



# def calculate_metric_percase(pred,gt):
#     pred[pred>0]=1
#     gt[gt>0]=1
#     if pred.sum()>0 and gt.sum()>0:
#         hd95=metric.binary.hd95(pred,gt)
#         return  hd95
#     elif pred.sum()>0 and gt.sum()==0:
#         return 1,0
#     else:
#         return 0,0
