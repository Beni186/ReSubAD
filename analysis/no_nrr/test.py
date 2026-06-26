"""Test the no-NRR ablation: ARSP directly on query features (no bank, no residual)."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_PRETRAINED,
    DEFAULT_DATASET,
    DEFAULT_IMAGE_SIZE,
)
from dataset import get_data  # noqa: E402
from eval import compute_image_metrics, compute_pixel_metrics  # noqa: E402
from logger import Logger  # noqa: E402
from AnomalyCLIP_lib import get_similarity_map  # noqa: E402
from AnomalyCLIP_lib import load as ac_load  # noqa: E402
from utils import get_transform  # noqa: E402

from arsp_model import DirectArspHead  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Test no-NRR ARSP (query-only) model")
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p.add_argument("--model", type=str, default=DEFAULT_CLIP_MODEL)
    p.add_argument("--pretrained", type=str, default=DEFAULT_CLIP_PRETRAINED)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--levels", type=int, nargs="+", default=None)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_path", type=str, default="./outputs_no_nrr")
    p.add_argument("--save_metrics", type=str, default="")
    p.add_argument("--sigma", type=int, default=10)
    p.add_argument("--pixel_weight", type=float, default=0.5)
    p.add_argument("--metric_image_size", type=int, default=0)
    p.add_argument("--max_test_batches", type=int, default=0)
    return p.parse_args()


def min_max_norm(arr):
    a_min, a_max = arr.min(), arr.max()
    return arr if a_max == a_min else (arr - a_min) / (a_max - a_min)


def compute_scores(clip_model, model, images, levels, image_size, dpam_layer, sigma):
    with torch.no_grad():
        image_features, patch_features_list = clip_model.encode_image(
            images, feature_list=levels, DPAM_layer=dpam_layer, proj_use=True
        )
        if image_features.dim() == 3:
            image_features = image_features.squeeze(1)
        img_cls = F.normalize(image_features, dim=-1)
        logits = torch.matmul(img_cls, model.get_image_embeds().t()) / 0.07
        img_probs = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        layer_scores, _ = model.forward_pixel(patch_features_list)

    layer_maps = []
    for scores in layer_scores:
        similarity_map = get_similarity_map(scores, (image_size, image_size))
        anomaly_map = similarity_map[..., 1] + 1 - similarity_map[..., 0]
        layer_maps.append(anomaly_map)
    mean_maps = torch.stack(layer_maps, dim=0).mean(dim=0) / 2.0
    pixel_maps = [gaussian_filter(mean_maps[i].detach().cpu().numpy(), sigma=sigma) for i in range(mean_maps.shape[0])]
    return img_probs, pixel_maps


def resize_metric_arrays(masks, maps, metric_image_size):
    if metric_image_size <= 0:
        return masks, maps
    masks_t = torch.from_numpy(np.asarray(masks)).unsqueeze(1).float()
    maps_t = torch.from_numpy(np.asarray(maps)).unsqueeze(1).float()
    masks_t = F.interpolate(masks_t, size=(metric_image_size, metric_image_size), mode="nearest")
    maps_t = F.interpolate(maps_t, size=(metric_image_size, metric_image_size), mode="bilinear", align_corners=False)
    masks_np = masks_t.squeeze(1).numpy()
    maps_np = maps_t.squeeze(1).numpy()
    masks_np[masks_np > 0.5] = 1
    masks_np[masks_np <= 0.5] = 0
    return masks_np, maps_np


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(save_dir / "test_no_nrr.log")
    logger.info(f"Loading checkpoint from {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    levels = args.levels or ckpt_args.get("levels", [18, 21, 24])
    dpam_layer = int(ckpt_args.get("dpam_layer", 20))

    load_name = args.model
    if args.pretrained and os.path.isfile(args.pretrained):
        load_name = args.pretrained
    design_details = {
        "trainer": "AnomalyCLIP_TR", "vision_depth": 0, "language_depth": 0,
        "vision_ctx": 0, "language_ctx": 0, "learnabel_text_embedding_length": 0,
    }
    clip_model, _ = ac_load(load_name, device=device, design_details=design_details)
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()
    clip_model.visual.DAPM_replace(DPAM_layer=dpam_layer)

    model = DirectArspHead(
        embed_dim=768,
        num_layers=len(levels),
        proj_dim=int(ckpt_args.get("proj_dim", 512)),
        subspace_dim=int(ckpt_args.get("subspace_dim", 32)),
        normalize_input=not bool(ckpt_args.get("no_normalize_input", False)),
        use_average_neighbor=not bool(ckpt_args.get("no_average_neighbor", False)),
    ).to(device)
    model.load_state_dict(checkpoint.get("custom_model", checkpoint))
    model.eval()

    preprocess, target_transform = get_transform(args)
    dataset_cls_names, dataset, _ = get_data(args.dataset, preprocess, target_transform, training=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger.info(f"Dataset: {args.dataset}, Classes: {len(dataset_cls_names)}, Levels: {levels}")

    class_results = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if args.max_test_batches > 0 and batch_idx >= args.max_test_batches:
                break
            images = batch["img"].to(device)
            labels = batch["anomaly"].numpy()
            masks = batch["img_mask"]
            masks[masks > 0.5] = 1
            masks[masks <= 0.5] = 0
            if masks.dim() == 4:
                masks = masks.squeeze(1)
            masks = masks.numpy()
            cls_names = batch["cls_name"]

            scores, maps = compute_scores(clip_model, model, images, levels, args.image_size, dpam_layer, args.sigma)
            masks, maps = resize_metric_arrays(masks, maps, args.metric_image_size)
            for i, cls_name in enumerate(cls_names):
                if cls_name not in class_results:
                    class_results[cls_name] = {"scores": [], "labels": [], "maps": [], "masks": []}
                class_results[cls_name]["scores"].append(scores[i])
                class_results[cls_name]["labels"].append(labels[i])
                class_results[cls_name]["maps"].append(maps[i])
                class_results[cls_name]["masks"].append(masks[i])

    image_metrics_list, pixel_metrics_list, class_metric_map = [], [], {}
    for cls_name, data in sorted(class_results.items()):
        cls_scores = np.array(data["scores"])
        cls_labels = np.array(data["labels"])
        cls_maps = np.array(data["maps"])
        cls_masks = np.array(data["masks"])
        norm_pixel_preds = min_max_norm(cls_maps)
        norm_image_preds = min_max_norm(cls_scores)
        pmax_pred = norm_pixel_preds.max(axis=(1, 2))
        final_scores = (1.0 - args.pixel_weight) * norm_image_preds + args.pixel_weight * pmax_pred
        img_met = compute_image_metrics(cls_labels, final_scores)
        pix_met = compute_pixel_metrics(cls_masks, cls_maps)
        image_metrics_list.append(img_met)
        pixel_metrics_list.append(pix_met)
        class_metric_map[cls_name] = {"image": img_met, "pixel": pix_met}
        logger.info(f"Class: {cls_name}")
        logger.info(f"  I-Auroc:{100 * img_met['auroc']:.2f}\tI-AP:{100 * img_met['ap']:.2f}\tI-F1:{100 * img_met['f1']:.2f}")
        logger.info(f"  P-Auroc:{100 * pix_met['auroc']:.2f}\tP-AP:{100 * pix_met['ap']:.2f}\tP-AUPRO:{100 * pix_met['aupro']:.2f}")

    avg_img = {k: np.mean([m[k] for m in image_metrics_list]) for k in image_metrics_list[0]}
    avg_pix = {k: np.mean([m[k] for m in pixel_metrics_list]) for k in pixel_metrics_list[0]}
    logger.info("==== Average Metrics ====")
    logger.info(
        f"I-Auroc:{100 * avg_img['auroc']:.2f}\tI-AP:{100 * avg_img['ap']:.2f}\t"
        f"P-Auroc:{100 * avg_pix['auroc']:.2f}\tP-AP:{100 * avg_pix['ap']:.2f}\tP-AUPRO:{100 * avg_pix['aupro']:.2f}"
    )
    if args.save_metrics:
        out = Path(args.save_metrics)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"dataset": args.dataset, "checkpoint": args.checkpoint, "levels": levels,
                       "class_metrics": class_metric_map,
                       "average_metrics": {"image": avg_img, "pixel": avg_pix}}, f, indent=2)
        logger.info(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
