"""No-NRR ablation model: ARSP applied directly to the query feature.

This drops the entire Normal-Referenced Residualization stage -- no normal memory
bank and no query-to-normal residual. The subspace projection energy is computed on
the projected query feature q itself:

    q     = Norm(W_q f)                  (unit-normalized projected patch feature)
    e     = || U^T q ||^2 / ||q||^2      (= || U^T q ||^2, since ||q|| = 1)
    score = [1 - e, e]

The learnable parts are the per-layer query projections W_q, the per-layer subspace
bases (orthonormalized by QR at forward), and the two image-level prototypes.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.subspace import average_neighbor  # noqa: E402  (reuse multi-scale neighbor pooling)


class DirectSubspacePixelHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 3,
        proj_dim: int = 512,
        subspace_dim: int = 32,
        normalize_input: bool = True,
        use_average_neighbor: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.subspace_dim = subspace_dim
        self.normalize_input = normalize_input
        self.use_average_neighbor = use_average_neighbor
        self.eps = float(eps)

        self.query_proj = nn.ModuleList([nn.Linear(embed_dim, proj_dim, bias=False) for _ in range(num_layers)])
        self.bases = nn.ParameterList(
            [nn.Parameter(torch.empty(proj_dim, subspace_dim)) for _ in range(num_layers)]
        )
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

    def forward_layer(self, patch_feat_raw: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, Dict]:
        query = self._project_query(patch_feat_raw, layer_idx)
        basis = self._basis(layer_idx)
        projected = query @ basis
        numerator = projected.pow(2).sum(dim=-1)
        denominator = query.pow(2).sum(dim=-1).clamp_min(self.eps)  # ==1 for unit query
        energy = (numerator / denominator).clamp(0.0, 1.0)
        scores = torch.stack([1.0 - energy, energy], dim=-1).clamp(self.eps, 1.0 - self.eps)
        aux = {"energy": energy, "query": query}
        return scores, aux

    def forward(self, patch_features_list: List[torch.Tensor]):
        layer_scores, layer_aux = [], []
        for layer_idx, patch_feat in enumerate(patch_features_list):
            scores, aux = self.forward_layer(patch_feat, layer_idx)
            layer_scores.append(scores)
            layer_aux.append(aux)
        return layer_scores, layer_aux


class DirectArspHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 3,
        proj_dim: int = 512,
        subspace_dim: int = 32,
        normalize_input: bool = True,
        use_average_neighbor: bool = True,
    ):
        super().__init__()
        self.image_normal_embed = nn.Parameter(torch.empty(1, embed_dim))
        self.image_anomaly_embed = nn.Parameter(torch.empty(1, embed_dim))
        nn.init.normal_(self.image_normal_embed, std=0.02)
        nn.init.normal_(self.image_anomaly_embed, std=0.02)
        self.pixel_head = DirectSubspacePixelHead(
            embed_dim=embed_dim,
            num_layers=num_layers,
            proj_dim=proj_dim,
            subspace_dim=subspace_dim,
            normalize_input=normalize_input,
            use_average_neighbor=use_average_neighbor,
        )

    def get_image_embeds(self):
        normal = F.normalize(self.image_normal_embed, dim=-1)
        anomaly = F.normalize(self.image_anomaly_embed, dim=-1)
        return torch.cat([normal, anomaly], dim=0)

    def orthogonality_loss(self):
        return self.pixel_head.orthogonality_loss()

    def forward_pixel(self, patch_features_list: List[torch.Tensor], *args, **kwargs):
        # *args/**kwargs absorb anchor_bank/sample_class_ids so this is drop-in compatible.
        return self.pixel_head(patch_features_list)
