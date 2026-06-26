"""Residual representation analysis: q-space (raw query) vs r-space (residual).

For a chosen feature level (default the 24th layer), this script extracts, over a
test dataset, the projected query feature ``q`` (the input to residualization) and
the query-to-normal residual ``r = q - n``. It then:

  * fits a UMAP embedding for the q-space and the r-space separately, each rendered
    with two colorings -- (i) normal vs. anomaly and (ii) dataset/class;
  * reports the Fisher (trace-ratio) separability between normal and anomaly
    patches in each space, so the "category normal explained away, anomaly
    preserved" claim can be read off quantitatively.

Example (analyze MVTec test with a VisA-trained checkpoint):

    python analysis/residual/residual_umap.py \
        --checkpoint ckpt/visa/subspace/best_epoch.pth \
        --dataset mvtec --layer 24
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_CLIP_MODEL, DEFAULT_CLIP_PRETRAINED, DEFAULT_IMAGE_SIZE  # noqa: E402
from dataset import get_data  # noqa: E402
from AnomalyCLIP_lib import load as ac_load  # noqa: E402
from model.clip_utils import build_mask_transform  # noqa: E402
from model.subspace import (  # noqa: E402
    SubspaceAnomalyHead,
    class_names_to_ids,
    mask_to_patch_labels,
)
from utils import get_transform  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="q-space vs r-space residual analysis")
    p.add_argument("--checkpoint", type=str, default="ckpt/visa/subspace/best_epoch.pth")
    p.add_argument("--dataset", type=str, default="mvtec")
    p.add_argument("--layer", type=int, default=24, help="Feature level to analyze (must be in ckpt levels).")
    p.add_argument("--model", type=str, default=DEFAULT_CLIP_MODEL)
    p.add_argument("--pretrained", type=str, default=DEFAULT_CLIP_PRETRAINED)
    p.add_argument("--image_size", type=int, default=0, help="0 -> use checkpoint/default image size.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--reference_policy", type=str, default="", help="Override ckpt reference policy.")
    p.add_argument("--save_dir", type=str, default="analysis/residual/output")
    # sampling controls (keep UMAP tractable while preserving rare anomaly patches)
    p.add_argument("--max_images", type=int, default=300)
    p.add_argument("--normals_per_image", type=int, default=200)
    p.add_argument("--anomalies_per_image", type=int, default=400)
    p.add_argument("--max_points", type=int, default=30000)
    p.add_argument("--anomaly_fraction", type=float, default=0.4, help="Max share of plotted points that are anomalies.")
    # umap controls
    p.add_argument("--umap_neighbors", type=int, default=30)
    p.add_argument("--umap_min_dist", type=float, default=0.1)
    p.add_argument("--umap_metric", type=str, default="cosine")
    p.add_argument("--seed", type=int, default=111)
    return p.parse_args()


def fisher_trace_ratio(features: np.ndarray, labels: np.ndarray) -> float:
    """tr(S_b) / tr(S_w) between groups. Scale-invariant within a space; higher = more separable."""
    classes = np.unique(labels)
    global_mean = features.mean(axis=0, keepdims=True)
    s_b = 0.0
    s_w = 0.0
    for c in classes:
        mask = labels == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        x_c = features[mask]
        mu_c = x_c.mean(axis=0, keepdims=True)
        s_b += n_c * float(((mu_c - global_mean) ** 2).sum())
        s_w += float(((x_c - mu_c) ** 2).sum())
    return s_b / s_w if s_w > 0 else float("nan")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    levels = list(ckpt_args.get("levels", [18, 21, 24]))
    if args.layer not in levels:
        raise ValueError(f"--layer {args.layer} not in checkpoint levels {levels}.")
    layer_idx = levels.index(args.layer)
    dpam_layer = int(ckpt_args.get("dpam_layer", 20))
    reference_policy = args.reference_policy or ckpt_args.get("reference_policy", "all")
    image_size = args.image_size or int(ckpt_args.get("image_size", DEFAULT_IMAGE_SIZE))
    if "anchor_bank" not in checkpoint:
        raise RuntimeError("Checkpoint has no anchor_bank.")
    anchor_bank = checkpoint["anchor_bank"]
    class_to_id = checkpoint.get("class_to_id", anchor_bank.get("class_to_id", {}))

    print(f"[cfg] dataset={args.dataset} layer={args.layer} (idx {layer_idx}) "
          f"policy={reference_policy} image_size={image_size}")

    # ---- backbone + head ----
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
    clip_model.visual.DAPM_replace(DPAM_layer=dpam_layer)

    model = SubspaceAnomalyHead(
        embed_dim=768,
        num_layers=len(levels),
        proj_dim=int(ckpt_args.get("proj_dim", 512)),
        subspace_dim=int(ckpt_args.get("subspace_dim", 32)),
        normal_topk=int(ckpt_args.get("normal_topk", 5)),
        normal_temperature=float(ckpt_args.get("normal_temperature", 0.07)),
        normalize_input=not bool(ckpt_args.get("no_normalize_input", False)),
        use_average_neighbor=not bool(ckpt_args.get("no_average_neighbor", False)),
    ).to(device)
    state = checkpoint.get("custom_model", checkpoint.get("model", checkpoint))
    model.load_state_dict(state)
    model.eval()
    pixel_head = model.pixel_head

    # projected memory keys for this layer (computed once)
    anchor_feats = anchor_bank["features"][layer_idx].float().to(device)
    anchor_cids = anchor_bank["class_ids"][layer_idx].to(device)
    with torch.no_grad():
        memory = pixel_head._project_memory(anchor_feats, layer_idx)

    # ---- data ----
    preprocess, target_transform = get_transform(SimpleNamespace(image_size=image_size))
    target_transform = build_mask_transform(image_size)
    _, dataset, _ = get_data(args.dataset, preprocess, target_transform, training=False)
    # shuffle so a capped (--max_images) pass still spans many classes/defect types
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=gen
    )

    q_chunks, r_chunks, lbl_chunks, cls_chunks = [], [], [], []
    n_images = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="extract"):
            if n_images >= args.max_images:
                break
            images = batch["img"].to(device)
            masks = batch["img_mask"].to(device)
            cls_names = list(batch["cls_name"])
            n_images += images.shape[0]

            _, patch_features_list = clip_model.encode_image(
                images, feature_list=levels, DPAM_layer=dpam_layer, proj_use=True
            )
            patch_feat_raw = patch_features_list[layer_idx].float()
            q = pixel_head._project_query(patch_feat_raw, layer_idx)  # (B, T, p)
            sample_cids = class_names_to_ids(cls_names, class_to_id, device)
            reference = pixel_head._normal_reference(
                q, memory, anchor_cids, sample_class_ids=sample_cids, reference_policy=reference_policy
            )
            r = q - reference  # (B, T, p)

            side = int(round(q.shape[1] ** 0.5))
            patch_labels = mask_to_patch_labels(masks, side, threshold=float(ckpt_args.get("mask_threshold", 0.5)))

            q_np = q.cpu().numpy()
            r_np = r.cpu().numpy()
            lbl_np = patch_labels.cpu().numpy()
            for b in range(q_np.shape[0]):
                lab = lbl_np[b]
                anom_idx = np.nonzero(lab == 1)[0]
                norm_idx = np.nonzero(lab == 0)[0]
                if anom_idx.size > args.anomalies_per_image:
                    anom_idx = rng.choice(anom_idx, args.anomalies_per_image, replace=False)
                if norm_idx.size > args.normals_per_image:
                    norm_idx = rng.choice(norm_idx, args.normals_per_image, replace=False)
                sel = np.concatenate([norm_idx, anom_idx])
                if sel.size == 0:
                    continue
                q_chunks.append(q_np[b, sel])
                r_chunks.append(r_np[b, sel])
                lbl_chunks.append(lab[sel])
                cls_chunks.append(np.array([cls_names[b]] * sel.size))

    q_all = np.concatenate(q_chunks, axis=0)
    r_all = np.concatenate(r_chunks, axis=0)
    lbl_all = np.concatenate(lbl_chunks, axis=0)
    cls_all = np.concatenate(cls_chunks, axis=0)
    print(f"[data] collected {q_all.shape[0]} patches "
          f"({int((lbl_all == 1).sum())} anomaly / {int((lbl_all == 0).sum())} normal)")

    # ---- global stratified subsample (cap anomalies so plot stays readable) ----
    anom_pos = np.nonzero(lbl_all == 1)[0]
    norm_pos = np.nonzero(lbl_all == 0)[0]
    max_anom = min(anom_pos.size, int(args.max_points * args.anomaly_fraction))
    if anom_pos.size > max_anom:
        anom_pos = rng.choice(anom_pos, max_anom, replace=False)
    max_norm = min(norm_pos.size, args.max_points - anom_pos.size)
    if norm_pos.size > max_norm:
        norm_pos = rng.choice(norm_pos, max_norm, replace=False)
    keep = rng.permutation(np.concatenate([norm_pos, anom_pos]))
    q_all, r_all, lbl_all, cls_all = q_all[keep], r_all[keep], lbl_all[keep], cls_all[keep]
    print(f"[data] plotting {q_all.shape[0]} patches "
          f"({int((lbl_all == 1).sum())} anomaly / {int((lbl_all == 0).sum())} normal)")

    # ---- Fisher separability (normal vs anomaly) ----
    metrics = {
        "dataset": args.dataset,
        "layer": args.layer,
        "reference_policy": reference_policy,
        "n_points": int(q_all.shape[0]),
        "n_anomaly": int((lbl_all == 1).sum()),
        "n_normal": int((lbl_all == 0).sum()),
        "fisher_normal_vs_anomaly": {
            "q_space": fisher_trace_ratio(q_all, lbl_all),
            "r_space": fisher_trace_ratio(r_all, lbl_all),
        },
        "fisher_class_separability": {
            "q_space": fisher_trace_ratio(q_all, cls_all),
            "r_space": fisher_trace_ratio(r_all, cls_all),
        },
    }
    fq = metrics["fisher_normal_vs_anomaly"]["q_space"]
    fr = metrics["fisher_normal_vs_anomaly"]["r_space"]
    cq = metrics["fisher_class_separability"]["q_space"]
    cr = metrics["fisher_class_separability"]["r_space"]
    def _ratio(a, b):
        return f"x{a / b:.2f}" if b else "n/a"

    print(f"[fisher] normal/anomaly  q={fq:.4f}  r={fr:.4f}  (r/q {_ratio(fr, fq)})")
    print(f"[fisher] class identity  q={cq:.4f}  r={cr:.4f}  (r/q {_ratio(cr, cq)})")
    with open(save_dir / f"fisher_{args.dataset}_L{args.layer}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- UMAP embeddings ----
    import umap

    def embed(x):
        reducer = umap.UMAP(
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.seed,
        )
        return reducer.fit_transform(x)

    print("[umap] fitting q-space ...")
    emb_q = embed(q_all)
    print("[umap] fitting r-space ...")
    emb_r = embed(r_all)

    # ---- plots ----
    classes = sorted(set(cls_all.tolist()))
    cmap = plt.get_cmap("tab20")
    cls_color = {c: cmap(i % 20) for i, c in enumerate(classes)}

    def scatter_anom(ax, emb, title):
        nm = lbl_all == 0
        am = lbl_all == 1
        ax.scatter(emb[nm, 0], emb[nm, 1], s=3, c="#3b6fb0", alpha=0.4, label="normal", linewidths=0)
        ax.scatter(emb[am, 0], emb[am, 1], s=3, c="#d1493f", alpha=0.6, label="anomaly", linewidths=0)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=3, loc="best", fontsize=8)

    def scatter_class(ax, emb, title):
        for c in classes:
            m = cls_all == c
            ax.scatter(emb[m, 0], emb[m, 1], s=3, color=cls_color[c], alpha=0.5, label=c, linewidths=0)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=3, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6, ncol=1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    scatter_anom(axes[0, 0], emb_q, f"q-space (raw query) | normal vs anomaly\nFisher={fq:.3f}")
    scatter_class(axes[0, 1], emb_q, f"q-space (raw query) | class\nFisher(class)={cq:.3f}")
    scatter_anom(axes[1, 0], emb_r, f"r-space (residual) | normal vs anomaly\nFisher={fr:.3f}")
    scatter_class(axes[1, 1], emb_r, f"r-space (residual) | class\nFisher(class)={cr:.3f}")
    fig.suptitle(f"{args.dataset} | layer {args.layer} | policy={reference_policy}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = save_dir / f"umap_{args.dataset}_L{args.layer}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[save] {out}")
    print(f"[save] {save_dir / f'fisher_{args.dataset}_L{args.layer}.json'}")


if __name__ == "__main__":
    main()
