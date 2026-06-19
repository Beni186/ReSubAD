from __future__ import annotations
from typing import Dict, Union, List
import numpy as np
import torch
from torchmetrics import AveragePrecision, AUROC, PrecisionRecallCurve
from anomalib.metrics.aupro import _AUPRO as AUPRO
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from skimage import measure

def _to_tensor(data: Union[np.ndarray, List, torch.Tensor]) -> torch.Tensor:
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data.float()

def _compute_best_f1(preds: torch.Tensor, targets: torch.Tensor, device: torch.device) -> float:
    """
    計算 Best F1。
    注意：如果 preds/targets 在 CPU，這裡的 device 應該傳入 'cpu'。
    """
    pr_curve = PrecisionRecallCurve(task="binary").to(device)
    precision, recall, thresholds = pr_curve(preds, targets)
    
    numerator = 2 * precision * recall
    denominator = precision + recall + 1e-10
    f1_scores = numerator / denominator
    return torch.max(f1_scores).item()

def compute_image_metrics(y_true, y_score) -> Dict[str, float]:
    labels = _to_tensor(y_true).int().cuda()
    preds = _to_tensor(y_score).cuda()
    device = preds.device

    metrics: Dict[str, float] = {}

    # 1. Image AUROC
    auroc_metric = AUROC(task="binary").to(device)
    auroc_metric.update(preds, labels)
    metrics["auroc"] = auroc_metric.compute().item()

    # 2. Image AP
    ap_metric = AveragePrecision(task="binary").to(device)
    ap_metric.update(preds, labels)
    metrics["ap"] = ap_metric.compute().item()

    # 3. Best F1 Score (取代 Anomalib F1Max)
    metrics["f1"] = _compute_best_f1(preds, labels, device)

    return metrics

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = np.min(amaps), np.max(amaps)
    if max_th <= min_th:
        return 0.0
    delta = (max_th - min_th) / max_step
    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[:] = amaps > th
        pro = []
        for i in range(len(amaps)):
            mask = masks[i]
            amap = binary_amaps[i]
            if np.sum(mask) == 0:
                continue  # Skip if no anomaly region
            labeled_mask = measure.label(mask)
            regions = measure.regionprops(labeled_mask)
            for region in regions:
                coords = region.coords
                tp = np.sum(amap[coords[:, 0], coords[:, 1]])
                pro.append(tp / region.area)
        avg_pro = np.mean(pro) if pro else 0
        pros.append(avg_pro)
        fp = np.sum(binary_amaps * (1 - masks))
        total_fp_pixels = np.sum(1 - masks)
        fpr = fp / total_fp_pixels if total_fp_pixels != 0 else 0
        fprs.append(fpr)
    pros, fprs = np.array(pros), np.array(fprs)
    valid = fprs <= expect_fpr
    if not np.any(valid):
        return 0.0
    fpr_valid = fprs[valid]
    pro_valid = pros[valid]
    if len(fpr_valid) < 2:
        return 0.0
    fpr_range = fpr_valid.max() - fpr_valid.min()
    if fpr_range <= 0:
        return 0.0
    fpr_norm = (fpr_valid - fpr_valid.min()) / fpr_range
    return auc(fpr_norm, pro_valid)

def compute_pixel_metrics(masks, maps) -> Dict[str, float]:
    # 1. 轉為 Tensor 並確保維度正確 (N, H, W)
    targets = _to_tensor(masks).int()
    preds = _to_tensor(maps).float()
    
    if preds.ndim == 4: preds = preds.squeeze(1)
    if targets.ndim == 4: targets = targets.squeeze(1)

    metrics: Dict[str, float] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. AUPRO (GPU 加速) ---
    # 只有在有異常點時計算才有意義
    if targets.sum() > 0:
        aupro_metric = AUPRO().to(device)
        # 僅將 AUPRO 需要的數據搬到 GPU
        aupro_metric.update(preds.to(device), targets.to(device))
        metrics["aupro"] = aupro_metric.compute().item()
        # metrics["aupro"] = cal_pro_score(targets, preds) if targets.sum() > 0 else 0
        del aupro_metric # 及時清理
    else:
        metrics["aupro"] = 0.0

    # --- 準備 Flatten 數據用於其他指標 (移至 CPU 避免 AUROC 排序時 OOM) ---
    preds_flat = preds.flatten().cpu()
    targets_flat = targets.flatten().cpu()
    
    # 釋放不再需要的空間
    del preds, targets

    # --- 2. Pixel AUROC & AP (CPU) ---
    # AUROC 在 CPU 算比較穩，因為它涉及全量像素排序
    auroc_m = AUROC(task="binary")
    metrics["auroc"] = auroc_m(preds_flat, targets_flat).item()

    ap_m = AveragePrecision(task="binary")
    metrics["ap"] = ap_m(preds_flat, targets_flat).item()

    # --- 3. Pixel Best F1 ---
    metrics["f1"] = _compute_best_f1(preds_flat, targets_flat, device='cpu')

    return metrics
