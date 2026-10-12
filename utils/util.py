import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
#
#     img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#     img = img / 255. if img.max() > 1.1 else img
#     if datasets == 'retinal':
#         msk = np.squeeze(msk, axis=0)
#         msk_pred = np.squeeze(msk_pred, axis=0)
#     else:
#         msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
#         msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)
#
#     plt.figure(figsize=(7, 15))
#
#     plt.subplot(3, 1, 1)
#     plt.imshow(img)
#     plt.axis('off')
#
#     plt.subplot(3, 1, 2)
#     plt.imshow(msk, cmap='gray')
#     plt.axis('off')
#
#     plt.subplot(3, 1, 3)
#     plt.imshow(msk_pred, cmap='gray')
#     plt.axis('off')
#
#     if test_data_name is not None:
#         save_path = save_path + test_data_name + '_'
#     plt.savefig(save_path + str(i) + '.png')
#     plt.close()

# def save_imgs(img, msk, msk_pred, i, save_path, datasets,
#               threshold=0.5, test_data_name=None):
#     # 1) 统一搬到 CPU → numpy
#     img_np   = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#     msk_np   = msk.squeeze(0).detach().cpu().numpy()
#     msk_pred_np = msk_pred.squeeze(0).detach().cpu().numpy()
#
#     # 2) 归一化/二值化
#     img_np = img_np / 255. if img_np.max() > 1.1 else img_np
#     if datasets == 'retinal':
#         pass  # 保持原值
#     else:
#         msk_np      = (msk_np > 0.5).astype(np.uint8)
#         msk_pred_np = (msk_pred_np > threshold).astype(np.uint8)
#
#     # 3) 画图
#     plt.figure(figsize=(7, 15))
#     plt.subplot(3, 1, 1)
#     plt.imshow(img_np)
#     plt.axis('off')
#
#     # plt.subplot(3, 1, 2)
#     # plt.imshow(msk_np, cmap='gray')
#     # plt.axis('off')
#
#     # 假设 msk_pred_np 是形状为 (1, 256, 256) 的 numpy 数组
#     msk_pred_np = msk_pred_np.squeeze()  # 变成 (256, 256)
#     plt.imshow(msk_pred_np, cmap='gray')
#     plt.axis('off')
#
#     plt.subplot(3, 1, 3)
#     plt.imshow(msk_pred_np, cmap='gray')
#     plt.axis('off')
#
#     # 4) 保存
#     if test_data_name is not None:
#         save_path = os.path.join(save_path, f"{test_data_name}_")
#     os.makedirs(save_path, exist_ok=True)
#     plt.savefig(os.path.join(save_path, f"{i}.png"))
#     plt.close()

def save_imgs(img, msk, msk_pred, i, save_path,
              datasets='other', threshold=0.5, test_data_name=None):
    """
    保存原图、GT、预测图的三栏可视化 PNG
    参数
    ----
    img      : Tensor (1,C,H,W)  0~1 float32 或 0~255 uint8
    msk      : Tensor (1,1,H,W)  0/1 或 0~1 float
    msk_pred : Tensor (1,1,H,W)  预测概率
    i        : 第几张图（用于文件名）
    save_path: 保存目录
    datasets : 'retinal' 保留原始多类，其他→二值
    threshold: 预测二值化阈值
    test_data_name : 可选前缀
    """

    # 1. 搬到 CPU → numpy
    img_np   = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    msk_np   = msk.squeeze(0).detach().cpu().numpy()
    msk_pred_np = msk_pred.squeeze(0).detach().cpu().numpy()

    # 2. 处理通道、数值范围
    # 2-1 灰度→RGB（可选）
    if img_np.ndim == 2:                       # (H,W)
        img_np = np.stack([img_np]*3, axis=-1)
    elif img_np.shape[-1] == 1:               # (H,W,1)
        img_np = np.repeat(img_np, 3, axis=-1)

    # 2-2 归一化到 0-1
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    # 2-3 转 0-255 uint8
    img_np = (img_np * 255).astype(np.uint8)

    # 3. 标签处理
    msk_np      = msk_np.squeeze()
    msk_pred_np = msk_pred_np.squeeze()

    if datasets != 'retinal':
        msk_np      = (msk_np > 0.5).astype(np.uint8)
        msk_pred_np = (msk_pred_np > threshold).astype(np.uint8)

    # 4. 画图
    plt.figure(figsize=(7, 15))

    plt.subplot(3, 1, 1)
    plt.imshow(img_np)
    plt.title('Image'); plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(msk_np, cmap='gray', vmin=0, vmax=1)
    plt.title('Ground Truth'); plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(msk_pred_np, cmap='gray', vmin=0, vmax=1)
    plt.title('Prediction'); plt.axis('off')

    # 5. 保存
    if test_data_name is not None:
        save_path = os.path.join(save_path, f"{test_data_name}_")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{i}.png"), bbox_inches='tight')
    plt.close()