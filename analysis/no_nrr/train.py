"""Train the no-NRR ablation: ARSP directly on query features (no bank, no residual)."""
import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
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
from AnomalyCLIP_lib import get_similarity_map  # noqa: E402
from AnomalyCLIP_lib import load as ac_load  # noqa: E402
from logger import Logger  # noqa: E402
from loss import BinaryDiceLoss, FocalLoss  # noqa: E402
from model.clip_utils import build_mask_transform  # noqa: E402
from model.subspace import mask_to_patch_labels  # noqa: E402
from utils import get_transform  # noqa: E402

from arsp_model import DirectArspHead  # noqa: E402


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="Train no-NRR ARSP (query-only) model")
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p.add_argument("--model", type=str, default=DEFAULT_CLIP_MODEL)
    p.add_argument("--pretrained", type=str, default=DEFAULT_CLIP_PRETRAINED)
    p.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--levels", type=int, nargs="+", default=[18, 21, 24])
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_path", type=str, default="./ckpt/no_nrr")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--dpam_layer", type=int, default=20)
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--subspace_dim", type=int, default=32)
    p.add_argument("--no_average_neighbor", action="store_true")
    p.add_argument("--no_normalize_input", action="store_true")
    p.add_argument("--mask_threshold", type=float, default=0.5)
    p.add_argument("--image_weight", type=float, default=1.0)
    p.add_argument("--seg_weight", type=float, default=1.0)
    p.add_argument("--energy_weight", type=float, default=0.5)
    p.add_argument("--orth_weight", type=float, default=0.01)
    p.add_argument("--energy_margin", type=float, default=0.5)
    p.add_argument("--max_train_batches", type=int, default=0, help="Debug only. 0 means full epoch.")
    return p.parse_args()


def compute_energy_loss(layer_aux, masks, energy_margin, mask_threshold):
    energy_losses = []
    for aux in layer_aux:
        energy = aux["energy"]
        side = int(round(energy.shape[1] ** 0.5))
        patch_labels = mask_to_patch_labels(masks, side, threshold=mask_threshold).to(energy.device)
        normal_mask = patch_labels == 0
        anomaly_mask = patch_labels == 1
        normal_energy = energy[normal_mask].mean() if bool(normal_mask.any()) else energy.mean() * 0.0
        if bool(anomaly_mask.any()):
            anomaly_loss = F.relu(energy_margin - energy[anomaly_mask]).mean()
        else:
            anomaly_loss = energy.mean() * 0.0
        energy_losses.append(normal_energy + anomaly_loss)
    return torch.stack(energy_losses).mean()


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(save_dir / "train_no_nrr.log")
    logger.info("==== Training no-NRR ARSP (query-only) ====")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

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
    clip_model.visual.DAPM_replace(DPAM_layer=args.dpam_layer)

    preprocess, _ = get_transform(args)
    target_transform = build_mask_transform(args.image_size)
    dataset_cls_names, train_dataset, dataset_root = get_data(args.dataset, preprocess, target_transform, training=True)
    class_to_id = {str(c): i for i, c in enumerate(dataset_cls_names)}

    model = DirectArspHead(
        embed_dim=768,
        num_layers=len(args.levels),
        proj_dim=args.proj_dim,
        subspace_dim=args.subspace_dim,
        normalize_input=not args.no_normalize_input,
        use_average_neighbor=not args.no_average_neighbor,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("custom_model", ckpt))
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    logger.info("Training query-only ARSP head...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = defaultdict(float)
        pbar = tqdm(train_loader)
        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            images = batch["img"].to(device)
            masks = batch["img_mask"].to(device)
            labels = batch["anomaly"].to(device).long()
            gt = masks.squeeze(1)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            optimizer.zero_grad()
            with torch.no_grad():
                image_features, patch_features_list = clip_model.encode_image(
                    images, feature_list=args.levels, DPAM_layer=args.dpam_layer, proj_use=True
                )
            if image_features.dim() == 3:
                image_features = image_features.squeeze(1)
            img_cls = F.normalize(image_features, dim=-1)
            logits = torch.matmul(img_cls, model.get_image_embeds().t()) / 0.07
            img_loss = F.cross_entropy(logits, labels)

            layer_scores, layer_aux = model.forward_pixel(patch_features_list)
            seg_loss = 0.0
            for scores in layer_scores:
                am = get_similarity_map(scores, (args.image_size, args.image_size)).permute(0, 3, 1, 2)
                seg_loss = seg_loss + loss_focal(am, gt.unsqueeze(1))
                seg_loss = seg_loss + loss_dice(am[:, 1, :, :], gt)
                seg_loss = seg_loss + loss_dice(am[:, 0, :, :], 1 - gt)
            seg_loss = seg_loss / len(layer_scores)

            energy_loss = compute_energy_loss(layer_aux, masks, args.energy_margin, args.mask_threshold)
            orth_loss = model.orthogonality_loss()

            loss = (
                args.image_weight * img_loss
                + args.seg_weight * seg_loss
                + args.energy_weight * energy_loss
                + args.orth_weight * orth_loss
            )
            loss.backward()
            optimizer.step()

            totals["loss"] += loss.item()
            totals["img"] += img_loss.item()
            totals["seg"] += seg_loss.item()
            totals["energy"] += energy_loss.item()
            totals["orth"] += orth_loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "seg": f"{seg_loss.item():.3f}",
                              "energy": f"{energy_loss.item():.3f}"})

        denom = max(args.max_train_batches if args.max_train_batches > 0 else len(train_loader), 1)
        logger.info(
            f"Epoch {epoch + 1} Avg Loss: {totals['loss'] / denom:.4f} "
            f"(Img: {totals['img'] / denom:.4f} | Seg: {totals['seg'] / denom:.4f} | "
            f"Energy: {totals['energy'] / denom:.4f} | Orth: {totals['orth'] / denom:.4f})"
        )
        checkpoint = {
            "custom_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": vars(args),
            "dataset_root": str(dataset_root),
            "class_to_id": class_to_id,
        }
        torch.save(checkpoint, save_dir / f"epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    main()
