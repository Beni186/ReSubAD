import argparse
import json
import os
import random
import sys
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
from loss import BinaryDiceLoss, BinaryFocalLoss, FocalLoss  # noqa: E402
from model.clip_utils import build_mask_transform  # noqa: E402
from model.subspace import (  # noqa: E402
    SubspaceAnomalyHead,
    build_normal_anchor_bank,
    class_names_to_ids,
    mask_to_patch_labels,
)
from utils import get_transform  # noqa: E402


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train residual subspace anomaly model")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=str, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--pretrained", type=str, default=DEFAULT_CLIP_PRETRAINED)
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--levels", type=int, nargs="+", default=[18, 21, 24])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="./ckpt/subspace")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--dpam_layer", type=int, default=20)

    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--subspace_dim", type=int, default=32)
    parser.add_argument("--normal_topk", type=int, default=5)
    parser.add_argument("--normal_temperature", type=float, default=0.07)
    parser.add_argument("--reference_policy", choices=["all", "same_class", "exclude_same_class"], default="all")
    parser.add_argument("--no_average_neighbor", action="store_true")
    parser.add_argument("--no_normalize_input", action="store_true")

    parser.add_argument("--normal_bank_source", choices=["normal_images", "all_normal_regions"], default="normal_images")
    parser.add_argument("--bank_reduce", choices=["kmeans", "reservoir", "coreset"], default="reservoir")
    parser.add_argument("--coreset_projection_dim", type=int, default=128)
    parser.add_argument("--coreset_distance", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--bank_candidates_per_class", type=int, default=4000)
    parser.add_argument("--bank_anchors_per_class", type=int, default=256)
    parser.add_argument("--mask_threshold", type=float, default=0.5)

    parser.add_argument("--image_weight", type=float, default=1.0)
    parser.add_argument("--image_fusion_weight", type=float, default=1.0)
    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument("--rec_weight", type=float, default=0.0)
    parser.add_argument("--energy_weight", type=float, default=0.5)
    parser.add_argument("--orth_weight", type=float, default=0.01)
    parser.add_argument("--energy_margin", type=float, default=0.5)

    parser.add_argument("--max_bank_batches", type=int, default=0, help="Debug only. 0 means full bank pass.")
    parser.add_argument("--max_train_batches", type=int, default=0, help="Debug only. 0 means full epoch.")
    return parser.parse_args()


def compute_subspace_losses(layer_aux, masks, energy_margin, mask_threshold):
    rec_losses = []
    energy_losses = []
    for aux in layer_aux:
        energy = aux["energy"]
        residual_sq = aux["residual_norm"].pow(2)
        side = int(round(energy.shape[1] ** 0.5))
        patch_labels = mask_to_patch_labels(masks, side, threshold=mask_threshold).to(energy.device)
        normal_mask = patch_labels == 0
        anomaly_mask = patch_labels == 1

        if bool(normal_mask.any()):
            rec_losses.append(residual_sq[normal_mask].mean())
            normal_energy = energy[normal_mask].mean()
        else:
            normal_energy = energy.mean() * 0.0
            rec_losses.append(energy.mean() * 0.0)

        if bool(anomaly_mask.any()):
            anomaly_loss = F.relu(energy_margin - energy[anomaly_mask]).mean()
        else:
            anomaly_loss = energy.mean() * 0.0
        energy_losses.append(normal_energy + anomaly_loss)

    return torch.stack(rec_losses).mean(), torch.stack(energy_losses).mean()


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(save_dir / "train_subspace.log")
    logger.info("==== Training Residual Subspace Model ====")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    load_name = args.model
    if args.pretrained and os.path.isfile(args.pretrained):
        load_name = args.pretrained

    design_details = {
        "trainer": "AnomalyCLIP_TR",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "learnabel_text_embedding_length": 0,
    }
    clip_model, _ = ac_load(load_name, device=device, design_details=design_details)
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()
    clip_model.visual.DAPM_replace(DPAM_layer=args.dpam_layer)

    preprocess, _ = get_transform(args)
    target_transform = build_mask_transform(args.image_size)
    dataset_cls_names, bank_dataset, dataset_root = get_data(
        args.dataset,
        preprocess,
        target_transform,
        training=False,
        split="test",
    )
    _, train_dataset, _ = get_data(
        args.dataset,
        preprocess,
        target_transform,
        training=True,
        split="test",
    )
    class_to_id = {str(cls_name): idx for idx, cls_name in enumerate(dataset_cls_names)}

    memory_bank_path = save_dir / "memory_bank.pth"
    if memory_bank_path.is_file():
        logger.info(f"Phase 1: loading normal anchor bank from {memory_bank_path}")
        anchor_bank = torch.load(memory_bank_path, map_location="cpu")
    else:
        bank_loader = DataLoader(
            bank_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        logger.info("Phase 1: building normal anchor bank...")
        anchor_bank = build_normal_anchor_bank(
            clip_model=clip_model,
            dataloader=bank_loader,
            levels=args.levels,
            class_to_id=class_to_id,
            device=device,
            candidates_per_class=args.bank_candidates_per_class,
            anchors_per_class=args.bank_anchors_per_class,
            bank_reduce=args.bank_reduce,
            coreset_projection_dim=args.coreset_projection_dim,
            coreset_distance=args.coreset_distance,
            source=args.normal_bank_source,
            mask_threshold=args.mask_threshold,
            dpam_layer=args.dpam_layer,
            use_average_neighbor=not args.no_average_neighbor,
            seed=args.seed,
            max_batches=args.max_bank_batches,
        )
        torch.save(anchor_bank, memory_bank_path)
        logger.info(f"Saved normal anchor bank to {memory_bank_path}")
    for layer, stats in zip(args.levels, anchor_bank["stats"]):
        total_anchors = sum(item["anchors"] for item in stats.values())
        logger.info(f"Layer {layer} normal anchors: {total_anchors}")

    model = SubspaceAnomalyHead(
        embed_dim=768,
        num_layers=len(args.levels),
        proj_dim=args.proj_dim,
        subspace_dim=args.subspace_dim,
        normal_topk=args.normal_topk,
        normal_temperature=args.normal_temperature,
        normalize_input=not args.no_normalize_input,
        use_average_neighbor=not args.no_average_neighbor,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    start_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        state = checkpoint.get("custom_model", checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
        model.load_state_dict(state)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_img_fusion = BinaryFocalLoss()

    logger.info("Phase 2: training P and U...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = defaultdict_float()
        pbar = tqdm(train_loader)

        for batch_idx, batch in enumerate(pbar):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

            images = batch["img"].to(device)
            masks = batch["img_mask"].to(device)
            labels = batch["anomaly"].to(device).long()
            sample_class_ids = class_names_to_ids(batch["cls_name"], class_to_id, device)

            gt = masks.squeeze(1)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            optimizer.zero_grad()
            with torch.no_grad():
                image_features, patch_features_list = clip_model.encode_image(
                    images,
                    feature_list=args.levels,
                    DPAM_layer=args.dpam_layer,
                    proj_use=True,
                )

            if image_features.dim() == 3:
                image_features = image_features.squeeze(1)
            img_cls = F.normalize(image_features, dim=-1)
            logits = torch.matmul(img_cls, model.get_image_embeds().t()) / 0.07
            img_loss = F.cross_entropy(logits, labels)

            layer_scores, layer_aux = model.forward_pixel(
                patch_features_list,
                anchor_bank,
                sample_class_ids=sample_class_ids,
                reference_policy=args.reference_policy,
            )

            seg_loss = 0.0
            map_max_scores = []
            for scores in layer_scores:
                map_max_scores.append(scores[:, :, 1].max(dim=1).values)
                am = get_similarity_map(scores, (args.image_size, args.image_size)).permute(0, 3, 1, 2)
                seg_loss = seg_loss + loss_focal(am, gt.unsqueeze(1))
                seg_loss = seg_loss + loss_dice(am[:, 1, :, :], gt)
                seg_loss = seg_loss + loss_dice(am[:, 0, :, :], 1 - gt)
            seg_loss = seg_loss / len(layer_scores)

            avg_map_max_score = sum(map_max_scores) / len(map_max_scores)
            img_probs = F.softmax(logits, dim=-1)[:, 1]
            img_fusion_score = 0.5 * (img_probs + avg_map_max_score)
            # img_fusion_loss = loss_img_fusion(img_fusion_score, labels.float())
            rec_loss, energy_loss = compute_subspace_losses(
                layer_aux,
                masks,
                energy_margin=args.energy_margin,
                mask_threshold=args.mask_threshold,
            )
            orth_loss = model.orthogonality_loss()

            loss = (
                args.image_weight * img_loss
                # + args.image_fusion_weight * img_fusion_loss
                + args.seg_weight * seg_loss
                + args.rec_weight * rec_loss
                + args.energy_weight * energy_loss
                + args.orth_weight * orth_loss
            )
            loss.backward()
            optimizer.step()

            totals["loss"] += loss.item()
            totals["img"] += img_loss.item()
            # totals["img2"] += img_fusion_loss.item()
            totals["img2"] += 0.0
            totals["seg"] += seg_loss.item()
            totals["rec"] += rec_loss.item()
            totals["energy"] += energy_loss.item()
            totals["orth"] += orth_loss.item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.3f}",
                    "seg": f"{seg_loss.item():.3f}",
                    "rec": f"{rec_loss.item():.3f}",
                    "energy": f"{energy_loss.item():.3f}",
                }
            )

        denom = args.max_train_batches if args.max_train_batches > 0 else len(train_loader)
        denom = max(denom, 1)
        logger.info(
            f"Epoch {epoch + 1} Avg Loss: {totals['loss'] / denom:.4f} "
            f"(Img: {totals['img'] / denom:.4f} | Img2: {totals['img2'] / denom:.4f} | "
            f"Seg: {totals['seg'] / denom:.4f} | Rec: {totals['rec'] / denom:.4f} | "
            f"Energy: {totals['energy'] / denom:.4f} | Orth: {totals['orth'] / denom:.4f})"
        )

        checkpoint = {
            "custom_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": vars(args),
            "dataset_root": str(dataset_root),
            "class_to_id": class_to_id,
            "anchor_bank": anchor_bank,
        }
        torch.save(checkpoint, save_dir / f"epoch_{epoch + 1}.pth")

    with open(save_dir / "anchor_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "dataset_root": str(dataset_root),
                "levels": args.levels,
                "normal_bank_source": args.normal_bank_source,
                "bank_reduce": args.bank_reduce,
                "coreset_projection_dim": args.coreset_projection_dim,
                "coreset_distance": args.coreset_distance,
                "stats": anchor_bank["stats"],
            },
            f,
            indent=2,
        )


def defaultdict_float():
    from collections import defaultdict

    return defaultdict(float)


if __name__ == "__main__":
    main()
