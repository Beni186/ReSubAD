import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def average_neighbor(x, kernels=(1, 3, 5)):
    batch_size, num_tokens, channels = x.shape
    side = int(round(num_tokens ** 0.5))
    if side * side != num_tokens:
        raise RuntimeError(f"Token sequence length {num_tokens} is not a square grid.")

    x_2d = x.transpose(1, 2).reshape(batch_size, channels, side, side)
    pooled = [
        F.avg_pool2d(x_2d, kernel_size=kernel, stride=1, padding=kernel // 2)
        for kernel in kernels
    ]
    return (sum(pooled) / len(pooled)).flatten(2).transpose(1, 2)


def mask_to_patch_labels(mask, side, threshold=0.5):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    patch_mask = F.interpolate(mask.float(), size=(side, side), mode="nearest")
    patch_mask = patch_mask.squeeze(1).reshape(mask.shape[0], side * side)
    return (patch_mask > threshold).long()


class TensorReservoir:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.rng = random.Random(seed)
        self.items = []
        self.seen = 0

    def add_many(self, features: torch.Tensor):
        features = features.detach().cpu()
        if self.capacity <= 0:
            self.items.append(features)
            self.seen += int(features.shape[0])
            return

        for feature in features:
            self.seen += 1
            if len(self.items) < self.capacity:
                self.items.append(feature.clone())
                continue
            replace_at = self.rng.randint(0, self.seen - 1)
            if replace_at < self.capacity:
                self.items[replace_at] = feature.clone()

    def tensor(self, feature_dim: int) -> torch.Tensor:
        if not self.items:
            return torch.empty(0, feature_dim)
        if self.capacity <= 0:
            return torch.cat(self.items, dim=0)
        return torch.stack(self.items, dim=0)


def _greedy_coreset_indices(
    features: torch.Tensor,
    target_size: int,
    seed: int,
    projection_dim: int = 128,
    distance: str = "euclidean",
) -> torch.Tensor:
    """PatchCore-style greedy k-center coreset selection.

    The optional random projection is used only for selecting indices. The caller
    should use the returned indices to gather the original high-dimensional
    features so the downstream trainable projection input dimension is unchanged.
    """
    if distance not in {"euclidean", "cosine"}:
        raise ValueError(f"Unsupported coreset distance: {distance}")

    n, dim = features.shape
    if target_size <= 0 or n <= target_size:
        return torch.arange(n, dtype=torch.long)

    x = features.detach().float().cpu()
    if projection_dim > 0 and projection_dim < dim:
        generator = torch.Generator().manual_seed(seed)
        random_matrix = torch.empty(dim, projection_dim, dtype=x.dtype)
        random_matrix.normal_(mean=0.0, std=1.0 / math.sqrt(float(projection_dim)), generator=generator)
        x = x @ random_matrix

    if distance == "cosine":
        x = F.normalize(x, dim=-1)

    generator = torch.Generator().manual_seed(seed)
    first_idx = int(torch.randint(0, n, (1,), generator=generator).item())
    selected = [first_idx]

    if distance == "euclidean":
        nearest_distance = (x - x[first_idx:first_idx + 1]).pow(2).sum(dim=1)
    else:
        nearest_distance = 1.0 - (x @ x[first_idx:first_idx + 1].t()).squeeze(1)

    nearest_distance[first_idx] = 0.0
    for _ in range(1, target_size):
        next_idx = int(torch.argmax(nearest_distance).item())
        selected.append(next_idx)

        if distance == "euclidean":
            new_distance = (x - x[next_idx:next_idx + 1]).pow(2).sum(dim=1)
        else:
            new_distance = 1.0 - (x @ x[next_idx:next_idx + 1].t()).squeeze(1)

        nearest_distance = torch.minimum(nearest_distance, new_distance)
        nearest_distance[next_idx] = 0.0

    return torch.tensor(selected, dtype=torch.long)


def reduce_features(
    features: torch.Tensor,
    target_size: int,
    mode: str,
    seed: int,
    coreset_projection_dim: int = 128,
    coreset_distance: str = "euclidean",
) -> torch.Tensor:
    if target_size <= 0 or features.shape[0] <= target_size:
        return features
    if mode == "reservoir":
        perm = torch.randperm(features.shape[0], generator=torch.Generator().manual_seed(seed))
        return features[perm[:target_size]]
    if mode == "coreset":
        indices = _greedy_coreset_indices(
            features,
            target_size=target_size,
            seed=seed,
            projection_dim=coreset_projection_dim,
            distance=coreset_distance,
        )
        return F.normalize(features[indices], dim=-1)
    if mode != "kmeans":
        raise ValueError(f"Unsupported bank_reduce mode: {mode}")

    from sklearn.cluster import MiniBatchKMeans

    arr = features.numpy().astype(np.float32)
    kmeans = MiniBatchKMeans(
        n_clusters=target_size,
        random_state=seed,
        batch_size=min(max(target_size * 4, 1024), max(arr.shape[0], 1024)),
        n_init=3,
    )
    kmeans.fit(arr)
    centers = torch.from_numpy(kmeans.cluster_centers_).to(dtype=features.dtype)
    return F.normalize(centers, dim=-1)


def build_normal_anchor_bank(
    clip_model,
    dataloader,
    levels: List[int],
    class_to_id: Dict[str, int],
    device: torch.device,
    candidates_per_class: int = 4000,
    anchors_per_class: int = 256,
    bank_reduce: str = "kmeans",
    coreset_projection_dim: int = 128,
    coreset_distance: str = "euclidean",
    source: str = "normal_images",
    mask_threshold: float = 0.5,
    dpam_layer: int = 20,
    use_average_neighbor: bool = True,
    seed: int = 111,
    max_batches: int = 0,
):
    if source not in {"normal_images", "all_normal_regions"}:
        raise ValueError(f"Unsupported normal bank source: {source}")

    layer_count = len(levels)
    class_count = len(class_to_id)
    reservoirs = [
        [
            TensorReservoir(candidates_per_class, seed + layer_idx * 1009 + class_idx * 9176)
            for class_idx in range(class_count)
        ]
        for layer_idx in range(layer_count)
    ]
    candidate_counts = [defaultdict(int) for _ in range(layer_count)]
    feature_dims = [None for _ in range(layer_count)]

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="build-normal-anchors")):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            images = batch["img"].to(device)
            masks = batch["img_mask"].to(device)
            labels = batch["anomaly"].to(device).long()
            cls_names = list(batch["cls_name"])
            _, patch_features_list = clip_model.encode_image(
                images,
                feature_list=levels,
                DPAM_layer=dpam_layer,
                proj_use=True,
            )

            for layer_idx, patch_feat_raw in enumerate(patch_features_list):
                patch_feat = patch_feat_raw[:, 1:, :]
                if use_average_neighbor:
                    patch_feat = average_neighbor(patch_feat)
                patch_feat = F.normalize(patch_feat, dim=-1)
                feature_dims[layer_idx] = int(patch_feat.shape[-1])
                side = int(round(patch_feat.shape[1] ** 0.5))
                patch_labels = mask_to_patch_labels(masks, side, threshold=mask_threshold)

                for sample_idx, cls_name in enumerate(cls_names):
                    if source == "normal_images" and int(labels[sample_idx].item()) != 0:
                        continue
                    class_id = class_to_id[str(cls_name)]
                    normal_idx = torch.nonzero(patch_labels[sample_idx] == 0, as_tuple=False).flatten()
                    candidate_counts[layer_idx][str(cls_name)] += int(normal_idx.numel())
                    if normal_idx.numel() == 0:
                        continue
                    reservoirs[layer_idx][class_id].add_many(patch_feat[sample_idx, normal_idx])

    features_by_layer = []
    class_ids_by_layer = []
    stats_by_layer = []
    id_to_class = {idx: name for name, idx in class_to_id.items()}
    for layer_idx in range(layer_count):
        feature_dim = feature_dims[layer_idx]
        if feature_dim is None:
            raise RuntimeError("No features were extracted while building normal anchors.")

        layer_features = []
        layer_class_ids = []
        layer_stats = {}
        for class_id in range(class_count):
            cls_name = id_to_class[class_id]
            candidates = reservoirs[layer_idx][class_id].tensor(feature_dim)
            if candidates.numel() > 0:
                anchors = reduce_features(
                    candidates,
                    target_size=anchors_per_class,
                    mode=bank_reduce,
                    seed=seed + layer_idx * 1009 + class_id * 9176 + 17,
                    coreset_projection_dim=coreset_projection_dim,
                    coreset_distance=coreset_distance,
                )
            else:
                anchors = torch.empty(0, feature_dim)
            layer_stats[cls_name] = {
                "candidate_normal_patches": int(candidate_counts[layer_idx][cls_name]),
                "candidate_sampled": int(candidates.shape[0]),
                "anchors": int(anchors.shape[0]),
            }
            if anchors.numel() == 0:
                continue
            layer_features.append(anchors)
            layer_class_ids.append(torch.full((anchors.shape[0],), class_id, dtype=torch.long))

        if not layer_features:
            raise RuntimeError(f"No normal anchors were collected for layer {levels[layer_idx]}.")
        features_by_layer.append(torch.cat(layer_features, dim=0).cpu())
        class_ids_by_layer.append(torch.cat(layer_class_ids, dim=0).cpu())
        stats_by_layer.append(layer_stats)

    return {
        "features": features_by_layer,
        "class_ids": class_ids_by_layer,
        "class_to_id": dict(class_to_id),
        "levels": list(levels),
        "source": source,
        "bank_reduce": bank_reduce,
        "coreset_projection_dim": int(coreset_projection_dim),
        "coreset_distance": str(coreset_distance),
        "candidates_per_class": int(candidates_per_class),
        "anchors_per_class": int(anchors_per_class),
        "mask_threshold": float(mask_threshold),
        "use_average_neighbor": bool(use_average_neighbor),
        "stats": stats_by_layer,
    }


class SubspacePixelHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 4,
        proj_dim: int = 512,
        subspace_dim: int = 32,
        normal_topk: int = 5,
        normal_temperature: float = 0.07,
        normalize_input: bool = True,
        use_average_neighbor: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.proj_dim = proj_dim
        self.subspace_dim = subspace_dim
        self.normal_topk = int(normal_topk)
        self.normal_temperature = float(normal_temperature)
        self.normalize_input = normalize_input
        self.use_average_neighbor = use_average_neighbor
        self.eps = float(eps)

        self.query_proj = nn.ModuleList([nn.Linear(embed_dim, proj_dim, bias=False) for _ in range(num_layers)])
        self.memory_proj = nn.ModuleList([nn.Linear(embed_dim, proj_dim, bias=False) for _ in range(num_layers)])
        self.bases = nn.ParameterList(
            [nn.Parameter(torch.empty(proj_dim, subspace_dim)) for _ in range(num_layers)]
        )
        self._init_parameters()

    def _init_parameters(self):
        for basis in self.bases:
            nn.init.orthogonal_(basis)

    def _basis(self, layer_idx: int) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.bases[layer_idx], mode="reduced")
        return q[:, : self.subspace_dim]

    def orthogonality_loss(self) -> torch.Tensor:
        losses = []
        for basis in self.bases:
            norm_basis = F.normalize(basis, dim=0)
            gram = norm_basis.t() @ norm_basis
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            losses.append((gram - eye).pow(2).mean())
        return torch.stack(losses).mean()

    def _project_query(self, patch_feat_raw: torch.Tensor, layer_idx: int) -> torch.Tensor:
        patch_feat = patch_feat_raw[:, 1:, :]
        if self.use_average_neighbor:
            patch_feat = average_neighbor(patch_feat)
        if self.normalize_input:
            patch_feat = F.normalize(patch_feat, dim=-1)
        return F.normalize(self.query_proj[layer_idx](patch_feat), dim=-1)

    def _project_memory(self, memory_features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.normalize_input:
            memory_features = F.normalize(memory_features, dim=-1)
        return F.normalize(self.memory_proj[layer_idx](memory_features), dim=-1)

    def _normal_reference(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_class_ids: torch.Tensor,
        sample_class_ids: Optional[torch.Tensor],
        reference_policy: str,
    ) -> torch.Tensor:
        refs = []
        for batch_idx in range(query.shape[0]):
            if reference_policy == "exclude_same_class" and sample_class_ids is not None:
                valid = memory_class_ids != sample_class_ids[batch_idx]
                if not bool(valid.any()):
                    valid = torch.ones_like(memory_class_ids, dtype=torch.bool)
                curr_memory = memory[valid]
            elif reference_policy == "same_class" and sample_class_ids is not None:
                valid = memory_class_ids == sample_class_ids[batch_idx]
                if not bool(valid.any()):
                    valid = torch.ones_like(memory_class_ids, dtype=torch.bool)
                curr_memory = memory[valid]
            else:
                curr_memory = memory

            sims = query[batch_idx] @ curr_memory.t()
            k = min(max(self.normal_topk, 1), curr_memory.shape[0])
            top_values, top_indices = sims.topk(k=k, dim=-1)
            weights = F.softmax(top_values / self.normal_temperature, dim=-1)
            refs.append((weights.unsqueeze(-1) * curr_memory[top_indices]).sum(dim=1))
        return torch.stack(refs, dim=0)

    def forward_layer(
        self,
        patch_feat_raw: torch.Tensor,
        layer_idx: int,
        anchor_features: torch.Tensor,
        anchor_class_ids: torch.Tensor,
        sample_class_ids: Optional[torch.Tensor] = None,
        reference_policy: str = "all",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        query = self._project_query(patch_feat_raw, layer_idx)
        memory = self._project_memory(anchor_features, layer_idx)
        reference = self._normal_reference(
            query,
            memory,
            anchor_class_ids,
            sample_class_ids=sample_class_ids,
            reference_policy=reference_policy,
        )
        residual = query - reference
        basis = self._basis(layer_idx)
        projected = residual @ basis
        numerator = projected.pow(2).sum(dim=-1)
        denominator = residual.pow(2).sum(dim=-1).clamp_min(self.eps)
        energy = (numerator / denominator).clamp(0.0, 1.0)
        scores = torch.stack([1.0 - energy, energy], dim=-1).clamp(self.eps, 1.0 - self.eps)
        aux = {
            "energy": energy,
            "residual_norm": denominator.sqrt(),
            "query": query,
            "reference": reference,
        }
        return scores, aux

    def forward(
        self,
        patch_features_list: List[torch.Tensor],
        anchor_bank: Dict[str, List[torch.Tensor]],
        sample_class_ids: Optional[torch.Tensor] = None,
        reference_policy: str = "all",
    ):
        layer_scores = []
        layer_aux = []
        for layer_idx, patch_feat in enumerate(patch_features_list):
            anchors = anchor_bank["features"][layer_idx].to(
                device=patch_feat.device,
                dtype=patch_feat.dtype,
                non_blocking=True,
            )
            anchor_class_ids = anchor_bank["class_ids"][layer_idx].to(device=patch_feat.device, non_blocking=True)
            scores, aux = self.forward_layer(
                patch_feat,
                layer_idx,
                anchors,
                anchor_class_ids,
                sample_class_ids=sample_class_ids,
                reference_policy=reference_policy,
            )
            layer_scores.append(scores)
            layer_aux.append(aux)
        return layer_scores, layer_aux


class SubspaceAnomalyHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 4,
        proj_dim: int = 512,
        subspace_dim: int = 32,
        normal_topk: int = 5,
        normal_temperature: float = 0.07,
        normalize_input: bool = True,
        use_average_neighbor: bool = True,
    ):
        super().__init__()
        self.image_normal_embed = nn.Parameter(torch.empty(1, embed_dim))
        self.image_anomaly_embed = nn.Parameter(torch.empty(1, embed_dim))
        nn.init.normal_(self.image_normal_embed, std=0.02)
        nn.init.normal_(self.image_anomaly_embed, std=0.02)
        self.pixel_head = SubspacePixelHead(
            embed_dim=embed_dim,
            num_layers=num_layers,
            proj_dim=proj_dim,
            subspace_dim=subspace_dim,
            normal_topk=normal_topk,
            normal_temperature=normal_temperature,
            normalize_input=normalize_input,
            use_average_neighbor=use_average_neighbor,
        )

    def get_image_embeds(self):
        normal = F.normalize(self.image_normal_embed, dim=-1)
        anomaly = F.normalize(self.image_anomaly_embed, dim=-1)
        return torch.cat([normal, anomaly], dim=0)

    def orthogonality_loss(self):
        return self.pixel_head.orthogonality_loss()

    def forward_pixel(
        self,
        patch_features_list: List[torch.Tensor],
        anchor_bank: Dict[str, List[torch.Tensor]],
        sample_class_ids: Optional[torch.Tensor] = None,
        reference_policy: str = "all",
    ):
        return self.pixel_head(
            patch_features_list,
            anchor_bank,
            sample_class_ids=sample_class_ids,
            reference_policy=reference_policy,
        )


def class_names_to_ids(cls_names: Iterable[str], class_to_id: Dict[str, int], device: torch.device):
    return torch.tensor([class_to_id.get(str(name), -1) for name in cls_names], dtype=torch.long, device=device)
