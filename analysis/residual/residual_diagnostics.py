"""Diagnose where the residual signal lives.

The UMAP/Fisher analysis shows r-space and q-space have near-identical directional
structure (r/q ~ 1.0). This script tests the alternative hypothesis: the residual
encodes anomaly evidence in its *norm/energy* (how well a patch is explained by the
normal memory), not in a class-specific direction it removes.

Per patch it measures cos(q, n), ||r||, subspace energy e, and raw memory distance,
then reports the patch-level normal-vs-anomaly AUROC of each, plus a *normal-only*
class-separability Fisher for q vs r (the cleaner test of "normal component removed").
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_CLIP_MODEL, DEFAULT_CLIP_PRETRAINED, DEFAULT_IMAGE_SIZE  # noqa: E402
from dataset import get_data  # noqa: E402
from AnomalyCLIP_lib import load as ac_load  # noqa: E402
from model.clip_utils import build_mask_transform  # noqa: E402
from model.subspace import SubspaceAnomalyHead, class_names_to_ids, mask_to_patch_labels  # noqa: E402
from utils import get_transform  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="ckpt/visa/subspace/best_epoch.pth")
    p.add_argument("--dataset", type=str, default="mvtec")
    p.add_argument("--layers", type=int, nargs="+", default=[18, 21, 24])
    p.add_argument("--model", type=str, default=DEFAULT_CLIP_MODEL)
    p.add_argument("--pretrained", type=str, default=DEFAULT_CLIP_PRETRAINED)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_images", type=int, default=200)
    p.add_argument("--normals_per_image", type=int, default=200)
    p.add_argument("--anomalies_per_image", type=int, default=400)
    p.add_argument("--save_dir", type=str, default="analysis/residual/output")
    p.add_argument("--seed", type=int, default=111)
    return p.parse_args()


def fisher_trace_ratio(x, y):
    classes = np.unique(y)
    gm = x.mean(0, keepdims=True)
    sb = sw = 0.0
    for c in classes:
        m = y == c
        n_c = int(m.sum())
        if n_c == 0:
            continue
        xc = x[m]
        mu = xc.mean(0, keepdims=True)
        sb += n_c * float(((mu - gm) ** 2).sum())
        sw += float(((xc - mu) ** 2).sum())
    return sb / sw if sw > 0 else float("nan")


def auroc(label_anom, score):
    if label_anom.min() == label_anom.max():
        return float("nan")
    return float(roc_auc_score(label_anom, score))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cargs = ckpt.get("args", {})
    levels = list(cargs.get("levels", [18, 21, 24]))
    dpam_layer = int(cargs.get("dpam_layer", 20))
    policy = cargs.get("reference_policy", "all")
    image_size = int(cargs.get("image_size", DEFAULT_IMAGE_SIZE))
    eps = 1e-6
    anchor_bank = ckpt["anchor_bank"]
    class_to_id = ckpt.get("class_to_id", anchor_bank.get("class_to_id", {}))

    load_name = args.model
    if args.pretrained and os.path.isfile(args.pretrained):
        load_name = args.pretrained
    design = {"trainer": "AnomalyCLIP_TR", "vision_depth": 0, "language_depth": 0,
              "vision_ctx": 0, "language_ctx": 0, "learnabel_text_embedding_length": 0}
    clip_model, _ = ac_load(load_name, device=device, design_details=design)
    clip_model.eval()
    for p_ in clip_model.parameters():
        p_.requires_grad = False
    clip_model.visual.DAPM_replace(DPAM_layer=dpam_layer)

    model = SubspaceAnomalyHead(
        embed_dim=768, num_layers=len(levels),
        proj_dim=int(cargs.get("proj_dim", 512)), subspace_dim=int(cargs.get("subspace_dim", 32)),
        normal_topk=int(cargs.get("normal_topk", 5)), normal_temperature=float(cargs.get("normal_temperature", 0.07)),
        normalize_input=not bool(cargs.get("no_normalize_input", False)),
        use_average_neighbor=not bool(cargs.get("no_average_neighbor", False)),
    ).to(device)
    model.load_state_dict(ckpt.get("custom_model", ckpt.get("model", ckpt)))
    model.eval()
    ph = model.pixel_head

    preprocess, _ = get_transform(SimpleNamespace(image_size=image_size))
    target_transform = build_mask_transform(image_size)
    _, dataset, _ = get_data(args.dataset, preprocess, target_transform, training=False)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=gen)

    report = {}
    for layer in args.layers:
        lidx = levels.index(layer)
        anchor_feats = anchor_bank["features"][lidx].float().to(device)
        anchor_cids = anchor_bank["class_ids"][lidx].to(device)
        with torch.no_grad():
            memory = ph._project_memory(anchor_feats, lidx)
            U = ph._basis(lidx)  # (p, s) orthonormal

        acc = {k: [] for k in ["cos_qn", "rnorm", "energy", "rawdist", "label"]}
        q_n, r_n, cls_n = [], [], []  # normal-only, for class Fisher
        n_img = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"L{layer}"):
                if n_img >= args.max_images:
                    break
                images = batch["img"].to(device)
                masks = batch["img_mask"].to(device)
                cls_names = list(batch["cls_name"])
                n_img += images.shape[0]
                _, pfl = clip_model.encode_image(images, feature_list=levels, DPAM_layer=dpam_layer, proj_use=True)
                q = ph._project_query(pfl[lidx].float(), lidx)  # (B,T,p), unit norm
                scids = class_names_to_ids(cls_names, class_to_id, device)
                n = ph._normal_reference(q, memory, anchor_cids, sample_class_ids=scids, reference_policy=policy)
                r = q - n
                rnorm = r.norm(dim=-1)                                   # ||r||
                cos_qn = (q * n).sum(-1) / n.norm(dim=-1).clamp_min(eps)  # q unit
                proj = torch.matmul(r, U)                                # (B,T,s)
                energy = proj.pow(2).sum(-1) / rnorm.pow(2).clamp_min(eps)
                rawsim = torch.matmul(q, memory.t()).max(dim=-1).values   # best normal match
                rawdist = 1.0 - rawsim

                side = int(round(q.shape[1] ** 0.5))
                lab = mask_to_patch_labels(masks, side, threshold=float(cargs.get("mask_threshold", 0.5)))
                for b in range(q.shape[0]):
                    lb = lab[b].cpu().numpy()
                    a_idx = np.nonzero(lb == 1)[0]
                    nrm_idx = np.nonzero(lb == 0)[0]
                    if a_idx.size > args.anomalies_per_image:
                        a_idx = rng.choice(a_idx, args.anomalies_per_image, replace=False)
                    if nrm_idx.size > args.normals_per_image:
                        nrm_idx = rng.choice(nrm_idx, args.normals_per_image, replace=False)
                    sel = np.concatenate([nrm_idx, a_idx])
                    if sel.size == 0:
                        continue
                    acc["cos_qn"].append(cos_qn[b, sel].cpu().numpy())
                    acc["rnorm"].append(rnorm[b, sel].cpu().numpy())
                    acc["energy"].append(energy[b, sel].cpu().numpy())
                    acc["rawdist"].append(rawdist[b, sel].cpu().numpy())
                    acc["label"].append(lb[sel])
                    if nrm_idx.size:
                        q_n.append(q[b, nrm_idx].cpu().numpy())
                        r_n.append(r[b, nrm_idx].cpu().numpy())
                        cls_n.append(np.array([cls_names[b]] * nrm_idx.size))

        for k in acc:
            acc[k] = np.concatenate(acc[k])
        lab = acc["label"]
        q_n = np.concatenate(q_n); r_n = np.concatenate(r_n); cls_n = np.concatenate(cls_n)

        layer_rep = {
            "n_patches": int(lab.size),
            "cos_qn_mean": float(acc["cos_qn"].mean()),
            "cos_qn_normal": float(acc["cos_qn"][lab == 0].mean()),
            "cos_qn_anomaly": float(acc["cos_qn"][lab == 1].mean()),
            "rnorm_normal": float(acc["rnorm"][lab == 0].mean()),
            "rnorm_anomaly": float(acc["rnorm"][lab == 1].mean()),
            "auroc_rawdist": auroc(lab, acc["rawdist"]),
            "auroc_rnorm": auroc(lab, acc["rnorm"]),
            "auroc_energy": auroc(lab, acc["energy"]),
            "fisher_class_normalonly_q": fisher_trace_ratio(q_n, cls_n),
            "fisher_class_normalonly_r": fisher_trace_ratio(r_n, cls_n),
        }
        report[f"layer_{layer}"] = layer_rep
        print(f"\n=== layer {layer} ({layer_rep['n_patches']} patches) ===")
        print(f"  cos(q,n): all={layer_rep['cos_qn_mean']:.3f}  "
              f"normal={layer_rep['cos_qn_normal']:.3f}  anomaly={layer_rep['cos_qn_anomaly']:.3f}")
        print(f"  ||r||   : normal={layer_rep['rnorm_normal']:.3f}  anomaly={layer_rep['rnorm_anomaly']:.3f}")
        print(f"  AUROC (normal vs anomaly):  raw_mem_dist={layer_rep['auroc_rawdist']:.3f}  "
              f"||r||={layer_rep['auroc_rnorm']:.3f}  energy={layer_rep['auroc_energy']:.3f}")
        print(f"  class Fisher (NORMAL only):  q={layer_rep['fisher_class_normalonly_q']:.3f}  "
              f"r={layer_rep['fisher_class_normalonly_r']:.3f}")

    with open(save_dir / f"diagnostics_{args.dataset}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[save] {save_dir / f'diagnostics_{args.dataset}.json'}")


if __name__ == "__main__":
    main()
