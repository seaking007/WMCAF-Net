import random
import argparse
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader.dataset import MedicalDataSets
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize
import torch
from utils.util import AverageMeter
import utils.losses as losses
from utils.metrics import iou_score

# 新增：稳定计算 HD95 和 ASSD 所需
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure
)


from WMCAF import WMCAF


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_torch(41)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    default='WMCAF',
    choices=[
        "WMCAF",

    ],
    help='model'
)

parser.add_argument(
    '--base_dir',
    type=str,
    default="/home/robot/seaking007/code/my_mamba/data/busi/",
    help='dir'
)

parser.add_argument(
    '--test_base_dir',
    type=str,
    default="/home/robot/seaking007/code/my_mamba/data/busi/",
    help='dir'
)

parser.add_argument(
    '--train_file_dir',
    type=str,
    default="busi_train.txt",
    help='dir'
)

parser.add_argument(
    '--val_file_dir',
    type=str,
    default="busi_val.txt",
    help='dir'
)

parser.add_argument(
    '--test_file_dir',
    type=str,
    default="busi_test.txt",
    help='dir'
)

parser.add_argument(
    '--base_lr',
    type=float,
    default= 0.0003,
    # 0.0003
    help='segmentation network learning rate'
    #0.0001
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=8,
    help='batch_size per gpu'
)

parser.add_argument('--epoch', type=int, default=200)

parser.add_argument(
    '--logs',
    type=str,
    default="/home/robot/seaking007/code/my_mamba/logs_7_2_1",
    help='dir'
)


# 新增：HD95 和 ASSD 的二值化阈值
parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help='threshold used to binarize prediction for HD95 and ASSD'
)

# 新增：是否打印空 mask 样本
parser.add_argument(
    '--print_empty_mask',
    action='store_true',
    help='print samples whose prediction or label is empty'
)

args = parser.parse_args()

SAVE_ROOT = "/home/robot/seaking007/code/my_mamba/runs"
os.makedirs(SAVE_ROOT, exist_ok=True)

from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    Resize,
    Normalize
)


def getDataloader():
    img_size = 256

    train_transform = Compose([
        RandomRotate90(),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Resize(img_size, img_size),
        Normalize(),
    ])

    val_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])

    test_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])

    db_train = MedicalDataSets(
        base_dir=args.base_dir,
        split="train",
        transform=train_transform,
        train_file_dir=args.train_file_dir,
        val_file_dir=args.val_file_dir,
        test_file_dir=args.test_file_dir
    )

    db_val = MedicalDataSets(
        base_dir=args.base_dir,
        split="val",
        transform=val_transform,
        train_file_dir=args.train_file_dir,
        val_file_dir=args.val_file_dir,
        test_file_dir=args.test_file_dir
    )

    db_test = MedicalDataSets(
        base_dir=args.test_base_dir,
        split="test",
        transform=test_transform,
        train_file_dir=args.train_file_dir,
        val_file_dir=args.val_file_dir,
        test_file_dir=args.test_file_dir
    )

    print(
        "train num:{}, val num:{}, test num:{}".format(
            len(db_train),
            len(db_val),
            len(db_test)
        )
    )

    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=False
    )

    valloader = DataLoader(
        db_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12
    )

    testloader = DataLoader(
        db_test,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    return trainloader, valloader, testloader


def get_model(args):
    if args.model == "WMCAF":
        model = WMCAF()

    else:
        model = None
        print("model err")
        exit(0)

    return model.cuda()


from datetime import datetime
import logging

log_file_to = None


# =========================================================
# 新增：HD95 / ASSD 工具函数
# =========================================================

def to_float(value):
    """
    将 Tensor、NumPy 标量或 Python 数值转换为 float。
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()

    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])

    return float(value)


def output_to_binary_mask(output, threshold=0.5):
    """
    将模型输出转换成二值 mask。

    如果输出范围不在 [0, 1]，则认为是 logits，先执行 sigmoid。
    如果已经在 [0, 1]，则直接使用阈值进行二值化。
    """
    output = output.detach()

    output_min = output.min().item()
    output_max = output.max().item()

    if output_min < 0 or output_max > 1:
        output = torch.sigmoid(output)

    pred = (output > threshold).cpu().numpy().astype(bool)
    return pred


def target_to_binary_mask(target):
    """
    将标签转换成二值 mask。
    """
    target_np = target.detach().cpu().numpy()
    target_np = (target_np > 0.5).astype(bool)
    return target_np


def get_surface(mask):
    """
    提取二值 mask 的边界。
    """
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return mask

    structure = generate_binary_structure(mask.ndim, 1)

    eroded = binary_erosion(
        mask,
        structure=structure,
        border_value=0
    )

    surface = mask ^ eroded
    return surface


def get_surface_distances(pred, gt, spacing=None):
    """
    计算两个方向的表面距离：

    1. pred surface 到 gt surface
    2. gt surface 到 pred surface
    """
    pred_surface = get_surface(pred)
    gt_surface = get_surface(gt)

    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return None, None

    dt_gt = distance_transform_edt(
        ~gt_surface,
        sampling=spacing
    )
    dist_pred_to_gt = dt_gt[pred_surface]

    dt_pred = distance_transform_edt(
        ~pred_surface,
        sampling=spacing
    )
    dist_gt_to_pred = dt_pred[gt_surface]

    return dist_pred_to_gt, dist_gt_to_pred


def compute_hd95(pred, gt, spacing=None):
    """
    稳定版 HD95。

    空 mask 处理规则：
    1. pred 和 gt 都为空：返回 0.0
    2. 只有一方为空：返回 np.nan，最终平均时跳过
    3. 两方均有前景：正常计算 HD95
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    if pred_sum == 0 and gt_sum == 0:
        return 0.0

    if pred_sum == 0 or gt_sum == 0:
        return np.nan

    dist_pred_to_gt, dist_gt_to_pred = get_surface_distances(
        pred,
        gt,
        spacing=spacing
    )

    if dist_pred_to_gt is None or dist_gt_to_pred is None:
        return np.nan

    if dist_pred_to_gt.size == 0 or dist_gt_to_pred.size == 0:
        return np.nan

    all_surface_distances = np.concatenate([
        dist_pred_to_gt,
        dist_gt_to_pred
    ])

    if all_surface_distances.size == 0:
        return np.nan

    hd95 = np.percentile(all_surface_distances, 95)
    return float(hd95)


def compute_assd(pred, gt, spacing=None):
    """
    计算 ASSD。

    空 mask 处理规则：
    1. pred 和 gt 都为空：返回 0.0
    2. 只有一方为空：返回 np.nan，最终平均时跳过
    3. 两方均有前景：正常计算 ASSD
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    if pred_sum == 0 and gt_sum == 0:
        return 0.0

    if pred_sum == 0 or gt_sum == 0:
        return np.nan

    dist_pred_to_gt, dist_gt_to_pred = get_surface_distances(
        pred,
        gt,
        spacing=spacing
    )

    if dist_pred_to_gt is None or dist_gt_to_pred is None:
        return np.nan

    if dist_pred_to_gt.size == 0 or dist_gt_to_pred.size == 0:
        return np.nan

    assd = (
        dist_pred_to_gt.sum() + dist_gt_to_pred.sum()
    ) / (
        dist_pred_to_gt.size + dist_gt_to_pred.size
    )

    return float(assd)


def compute_batch_surface_metrics(
    output,
    target,
    threshold=0.5,
    spacing=None,
    print_empty_mask=False,
    batch_index=None
):
    """
    按 batch 同时计算 HD95 和 ASSD。

    返回：
    hd95_sum
    hd95_valid_count
    assd_sum
    assd_valid_count
    skipped_count
    """
    pred_np = output_to_binary_mask(
        output,
        threshold=threshold
    )
    target_np = target_to_binary_mask(target)

    hd95_sum = 0.0
    hd95_valid_count = 0

    assd_sum = 0.0
    assd_valid_count = 0

    skipped_count = 0

    for b in range(pred_np.shape[0]):
        pred_b = np.squeeze(pred_np[b])
        target_b = np.squeeze(target_np[b])

        if pred_b.ndim != 2 or target_b.ndim != 2:
            raise ValueError(
                "HD95/ASSD 当前按 2D mask 计算，"
                f"但得到 pred shape={pred_b.shape}, "
                f"target shape={target_b.shape}"
            )

        pred_pixels = int(pred_b.sum())
        gt_pixels = int(target_b.sum())

        hd95 = compute_hd95(
            pred_b,
            target_b,
            spacing=spacing
        )

        assd = compute_assd(
            pred_b,
            target_b,
            spacing=spacing
        )

        if np.isfinite(hd95):
            hd95_sum += hd95
            hd95_valid_count += 1

        if np.isfinite(assd):
            assd_sum += assd
            assd_valid_count += 1

        if not np.isfinite(hd95) or not np.isfinite(assd):
            skipped_count += 1

            if print_empty_mask:
                print(
                    f"[Empty Mask] batch={batch_index}, "
                    f"sample={b}, "
                    f"pred_pixels={pred_pixels}, "
                    f"gt_pixels={gt_pixels}"
                )

    return (
        hd95_sum,
        hd95_valid_count,
        assd_sum,
        assd_valid_count,
        skipped_count
    )


def train(args):
    global log_file_to

    log_dir = args.logs

    exp_name = (
        f"{args.model}_"
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    log_dir = os.path.join(SAVE_ROOT, exp_name)
    ckpt_dir = os.path.join(log_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)

    current_time = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    log_file = os.path.join(
        log_dir,
        f"{args.model}_train_{current_time}.log"
    )

    log_file_to = log_file

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
        ]
    )

    logging.info("Training process started.")
    logging.info(f"Model: {args.model}")
    logging.info(f"Base Directory: {args.base_dir}")
    logging.info(f"Train File Directory: {args.train_file_dir}")
    logging.info(f"Validation File Directory: {args.val_file_dir}")
    logging.info(f"Test File Directory: {args.test_file_dir}")
    logging.info(f"Learning Rate: {args.base_lr}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Maximum Epochs: {args.epoch}")
    logging.info(f"HD95/ASSD Threshold: {args.threshold}")

    logging.info("--------------------------------------------------")

    base_lr = args.base_lr
    trainloader, valloader, testloader = getDataloader()
    model = get_model(args)

    print(
        "train file dir:{} val file dir:{}".format(
            args.train_file_dir,
            args.val_file_dir
        )
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=0.0001
    )

    criterion = losses.__dict__['BCEDiceLoss']().cuda()

    print("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_loss = float('inf')
    best_model_name = None

    max_epoch = args.epoch
    max_iterations = len(trainloader) * max_epoch

    for epoch_num in range(max_epoch):
        model.train()

        avg_meters = {
            'loss': AverageMeter(),
            'iou': AverageMeter(),
            'train_dice': AverageMeter(),
            'val_loss': AverageMeter(),
            'val_iou': AverageMeter(),
            'SE': AverageMeter(),
            'PC': AverageMeter(),
            'F1': AverageMeter(),
            'ACC': AverageMeter(),
            'val_dice': AverageMeter(),
            'HD95': AverageMeter(),
        }

        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image']
            label_batch = sampled_batch['label']

            volume_batch = volume_batch.cuda()
            label_batch = label_batch.cuda()

            outputs = model(volume_batch)

            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            loss = criterion(outputs, label_batch)

            iou, dice, _, _, _, _, _, _ = iou_score(
                outputs,
                label_batch
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (
                1.0 - iter_num / max_iterations
            ) ** 0.9

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            avg_meters['loss'].update(
                loss.item(),
                volume_batch.size(0)
            )

            avg_meters['iou'].update(
                to_float(iou),
                volume_batch.size(0)
            )

            avg_meters['train_dice'].update(
                to_float(dice),
                volume_batch.size(0)
            )

        model.eval()

        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                input = sampled_batch['image']
                target = sampled_batch['label']

                input = input.cuda()
                target = target.cuda()

                output = model(input)

                if isinstance(output, (tuple, list)):
                    output = output[0]

                loss = criterion(output, target)

                iou, dice, SE, PC, F1, _, ACC, _ = iou_score(
                    output,
                    target
                )

                batch_size = input.size(0)

                avg_meters['val_loss'].update(
                    loss.item(),
                    batch_size
                )

                avg_meters['val_iou'].update(
                    to_float(iou),
                    batch_size
                )

                avg_meters['SE'].update(
                    to_float(SE),
                    batch_size
                )

                avg_meters['PC'].update(
                    to_float(PC),
                    batch_size
                )

                avg_meters['F1'].update(
                    to_float(F1),
                    batch_size
                )

                avg_meters['ACC'].update(
                    to_float(ACC),
                    batch_size
                )

                avg_meters['val_dice'].update(
                    to_float(dice),
                    batch_size
                )

        logging.info(
            f"Epoch [{epoch_num}/{max_epoch}] - "
            f"Train Loss: {avg_meters['loss'].avg:.4f}, "
            f"Train IoU: {avg_meters['iou'].avg:.4f}, "
            f"Train Dice: {avg_meters['train_dice'].avg:.4f}, "
            f"Val Loss: {avg_meters['val_loss'].avg:.4f}, "
            f"Val IoU: {avg_meters['val_iou'].avg:.4f}, "
            f"Val Dice: {avg_meters['val_dice'].avg:.4f}, "
            f"Val SE: {avg_meters['SE'].avg:.4f}, "
            f"Val PC: {avg_meters['PC'].avg:.4f}, "
            f"Val F1: {avg_meters['F1'].avg:.4f}, "
            f"Val ACC: {avg_meters['ACC'].avg:.4f}"
        )

        print(
            f"Epoch [{epoch_num}/{max_epoch}] - "
            f"Train Loss: {avg_meters['loss'].avg:.4f}, "
            f"Train IoU: {avg_meters['iou'].avg:.4f}, "
            f"Train Dice: {avg_meters['train_dice'].avg:.4f}, "
            f"Val Loss: {avg_meters['val_loss'].avg:.4f}, "
            f"Val IoU: {avg_meters['val_iou'].avg:.4f}, "
            f"Val Dice: {avg_meters['val_dice'].avg:.4f}, "
            f"Val SE: {avg_meters['SE'].avg:.4f}, "
            f"Val PC: {avg_meters['PC'].avg:.4f}, "
            f"Val F1: {avg_meters['F1'].avg:.4f}, "
            f"Val ACC: {avg_meters['ACC'].avg:.4f}"
        )

        # =========================
        # 保存验证集 loss 最低的模型
        # 不再使用 Early Stopping，训练会完整执行 max_epoch
        # =========================
        current_val_loss = avg_meters['val_loss'].avg

        if current_val_loss < best_loss:
            model_filename = os.path.join(
                ckpt_dir,
                (
                    f"{args.model}_model_"
                    f"{args.train_file_dir.split('.')[0]}.pth"
                )
            )

            best_model_name = model_filename

            torch.save(
                model.state_dict(),
                model_filename
            )

            best_loss = current_val_loss

            print(
                f"=> saved best model, "
                f"val_loss improved to {best_loss:.4f}"
            )

            logging.info(
                f"=> saved best model, "
                f"val_loss improved to {best_loss:.4f}"
            )

    if best_model_name is None:
        print(
            "No best model was saved. "
            "Please check validation loss or training process."
        )

        logging.info(
            "No best model was saved. "
            "Please check validation loss or training process."
        )

        return "Training stopped without saving a model."

    checkpoint_path = best_model_name
    print(checkpoint_path)

    if os.path.exists(checkpoint_path):
        print('#----------Testing----------#')

        best_weight = torch.load(
            checkpoint_path,
            map_location=torch.device('cpu')
        )

        model.load_state_dict(best_weight)
        model.eval()

        test_metrics = {
            'loss': 0.0,
            'iou': 0.0,
            'dice': 0.0,
            'SE': 0.0,
            'PC': 0.0,
            'F1': 0.0,
            'ACC': 0.0,
            'HD95': 0.0,
            'ASSD': 0.0,
        }

        hd95_valid_count = 0
        assd_valid_count = 0
        surface_skipped_count = 0

        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(testloader):
                input = sampled_batch['image']
                target = sampled_batch['label']

                input = input.cuda()
                target = target.cuda()

                output = model(input)

                if isinstance(output, (tuple, list)):
                    output = output[0]

                loss = criterion(output, target)

                # 修改点：
                # 不再使用 iou_score 返回的 HD95，避免空 mask 导致 inf。
                iou, dice, SE, PC, F1, _, ACC, _ = iou_score(
                    output,
                    target
                )

                (
                    hd95_sum,
                    hd95_count,
                    assd_sum,
                    assd_count,
                    skipped_count
                ) = compute_batch_surface_metrics(
                    output,
                    target,
                    threshold=args.threshold,
                    spacing=None,
                    print_empty_mask=args.print_empty_mask,
                    batch_index=i_batch
                )

                batch_size = input.size(0)

                test_metrics['loss'] += (
                    loss.item() * batch_size
                )

                test_metrics['iou'] += (
                    to_float(iou) * batch_size
                )

                test_metrics['dice'] += (
                    to_float(dice) * batch_size
                )

                test_metrics['SE'] += (
                    to_float(SE) * batch_size
                )

                test_metrics['PC'] += (
                    to_float(PC) * batch_size
                )

                test_metrics['F1'] += (
                    to_float(F1) * batch_size
                )

                test_metrics['ACC'] += (
                    to_float(ACC) * batch_size
                )

                test_metrics['HD95'] += hd95_sum
                test_metrics['ASSD'] += assd_sum

                hd95_valid_count += hd95_count
                assd_valid_count += assd_count
                surface_skipped_count += skipped_count

        total_samples = len(testloader.dataset)

        if total_samples == 0:
            raise RuntimeError("Test dataset is empty.")

        for key in [
            'loss',
            'iou',
            'dice',
            'SE',
            'PC',
            'F1',
            'ACC'
        ]:
            test_metrics[key] /= total_samples

        if hd95_valid_count > 0:
            test_metrics['HD95'] /= hd95_valid_count
        else:
            test_metrics['HD95'] = np.nan

        if assd_valid_count > 0:
            test_metrics['ASSD'] /= assd_valid_count
        else:
            test_metrics['ASSD'] = np.nan

        print('#----------Surface Distance Samples----------#')
        print(
            f"HD95 valid samples: "
            f"{hd95_valid_count}/{total_samples}"
        )
        print(
            f"ASSD valid samples: "
            f"{assd_valid_count}/{total_samples}"
        )
        print(
            f"Skipped empty-mask samples: "
            f"{surface_skipped_count}"
        )

        logging.info(
            f"HD95 valid samples: "
            f"{hd95_valid_count}/{total_samples}, "
            f"ASSD valid samples: "
            f"{assd_valid_count}/{total_samples}, "
            f"Skipped empty-mask samples: "
            f"{surface_skipped_count}"
        )

        print(
            f"Test Loss: {test_metrics['loss']:.4f}, "
            f"Test IoU: {test_metrics['iou']:.4f}, "
            f"Test Dice: {test_metrics['dice']:.4f}, "
            f"Test SE: {test_metrics['SE']:.4f}, "
            f"Test PC: {test_metrics['PC']:.4f}, "
            f"Test F1: {test_metrics['F1']:.4f}, "
            f"Test ACC: {test_metrics['ACC']:.4f}, "
            f"Test HD95: {test_metrics['HD95']:.4f}, "
            f"Test ASSD: {test_metrics['ASSD']:.4f}"
        )

        logging.info(
            f"Test Loss: {test_metrics['loss']:.4f}, "
            f"Test IoU: {test_metrics['iou']:.4f}, "
            f"Test Dice: {test_metrics['dice']:.4f}, "
            f"Test SE: {test_metrics['SE']:.4f}, "
            f"Test PC: {test_metrics['PC']:.4f}, "
            f"Test F1: {test_metrics['F1']:.4f}, "
            f"Test ACC: {test_metrics['ACC']:.4f}, "
            f"Test HD95: {test_metrics['HD95']:.4f}, "
            f"Test ASSD: {test_metrics['ASSD']:.4f}"
        )

        return "Training and Testing Finished!"

    else:
        print(
            f"Checkpoint file not found at {checkpoint_path}"
        )

        logging.info(
            f"Checkpoint file not found at {checkpoint_path}"
        )


import matplotlib.pyplot as plt
import re
import os


def plot_metrics_from_log(log_file):
    """
    解析日志文件并绘制指标折线图。
    """
    epochs = []
    train_loss = []
    train_iou = []
    train_dice = []
    val_loss = []
    val_iou = []
    val_dice = []
    val_se = []
    val_pc = []
    val_f1 = []
    val_acc = []

    with open(log_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            match = re.search(
                r"Epoch \[(\d+)/\d+\] - "
                r"Train Loss: ([\d.]+), "
                r"Train IoU: ([\d.]+), "
                r"Train Dice: ([\d.]+), "
                r"Val Loss: ([\d.]+), "
                r"Val IoU: ([\d.]+), "
                r"Val Dice: ([\d.]+), "
                r"Val SE: ([\d.]+), "
                r"Val PC: ([\d.]+), "
                r"Val F1: ([\d.]+), "
                r"Val ACC: ([\d.]+)",
                line
            )

            if match:
                (
                    epoch,
                    t_loss,
                    t_iou,
                    t_dice,
                    v_loss,
                    v_iou,
                    v_dice,
                    v_se,
                    v_pc,
                    v_f1,
                    v_acc
                ) = match.groups()

                epochs.append(int(epoch))
                train_loss.append(float(t_loss))
                train_iou.append(float(t_iou))
                train_dice.append(float(t_dice))
                val_loss.append(float(v_loss))
                val_iou.append(float(v_iou))
                val_dice.append(float(v_dice))
                val_se.append(float(v_se))
                val_pc.append(float(v_pc))
                val_f1.append(float(v_f1))
                val_acc.append(float(v_acc))

    if not epochs:
        print("No data extracted from log file. Exiting.")
        return

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(
        epochs,
        train_loss,
        label='Train Loss'
    )
    plt.plot(
        epochs,
        val_loss,
        label='Val Loss'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(
        epochs,
        train_iou,
        label='Train IoU'
    )
    plt.plot(
        epochs,
        val_iou,
        label='Val IoU'
    )
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('IoU')

    plt.subplot(2, 2, 3)
    plt.plot(
        epochs,
        train_dice,
        label='Train Dice',
        marker='o'
    )
    plt.plot(
        epochs,
        val_dice,
        label='Val Dice',
        marker='x'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.title('Dice')

    plt.subplot(2, 2, 4)
    plt.plot(
        epochs,
        val_se,
        label='Val SE',
        marker='o'
    )
    plt.plot(
        epochs,
        val_pc,
        label='Val PC',
        marker='x'
    )
    plt.plot(
        epochs,
        val_f1,
        label='Val F1',
        marker='s'
    )
    plt.plot(
        epochs,
        val_acc,
        label='Val ACC',
        marker='^'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    plt.title('Validation Metrics')

    output_path = os.path.join(
        os.path.dirname(log_file),
        "plots",
        "metrics.png"
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    plt.tight_layout()
    plt.savefig(output_path)

    print(
        f"Metrics plot saved to {output_path}"
    )

    plt.show()


if __name__ == "__main__":
    train(args)
    print(log_file_to)

    if log_file_to and os.path.exists(log_file_to):
        plot_metrics_from_log(log_file_to)
    else:
        print(
            f"日志文件未找到，请检查路径：{log_file_to}"
        )