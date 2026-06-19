from __future__ import annotations

from pathlib import Path
import math

import torch
from torchvision import transforms
import open_clip
import torch.nn.functional as F


def load_clip(model_name: str, image_size: int, pretrained: str = "openai", device: str | torch.device = "cuda"):
    """
    Load a CLIP backbone together with its default preprocessing pipeline.
    If `pretrained` points to an existing file, it will be loaded directly instead of downloading.
    """
    pretrained_source = Path(pretrained)
    if pretrained_source.is_file():
        pretrained_arg = str(pretrained_source.resolve())
    else:
        pretrained_arg = pretrained

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_arg,
    )
    
    # Resize positional embedding if needed
    resize_pos_embed(model, image_size)

    tokenizer = open_clip.get_tokenizer(model_name)
    # Adjust first two transforms to desired resolution, similar to AF-CLIP
    if hasattr(preprocess, "transforms") and len(preprocess.transforms) >= 2:
        preprocess.transforms[0] = transforms.Resize(size=(image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)
        preprocess.transforms[1] = transforms.CenterCrop(size=(image_size, image_size))

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, tokenizer


def build_mask_transform(image_size: int):
    """
    Resize masks to the CLIP resolution using nearest-neighbor interpolation and convert to tensor.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )


def encode_images(model, images: torch.Tensor) -> torch.Tensor:
    """
    Helper for encoding and normalizing image features with CLIP.
    """
    with torch.no_grad():
        feats, _ = encode_image_with_patches(model, images)
    return feats


def encode_image_with_patches(model, images: torch.Tensor):
    """
    Returns tuple(global_features, patch_tokens) before spatial pooling.
    """
    visual = model.visual
    dtype = visual.conv1.weight.dtype if hasattr(visual, "conv1") else images.dtype
    with torch.no_grad():
        if not hasattr(visual, "conv1"):
            raise NotImplementedError("Only ViT-style visual backbones are supported for patch extraction.")
        x = images.type(dtype)
        x = visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # NLC
        class_embedding = visual.class_embedding.to(x.dtype)
        class_embedding = class_embedding + torch.zeros(x.shape[0], 1, class_embedding.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([class_embedding, x], dim=1)
        positional_embedding = visual.positional_embedding.to(x.dtype)
        x = x + positional_embedding
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj
        global_feat = x[:, 0, :]
        patch_tokens = x[:, 1:, :]
        global_feat = F.normalize(global_feat, dim=-1)
    return global_feat, patch_tokens


def resize_pos_embed(model, image_size):
    vision_model = model.visual
    if not hasattr(vision_model, 'positional_embedding'):
        return

    # Determine patch size
    patch_size = None
    if hasattr(vision_model, 'patch_embed'):
        # timm style
        if hasattr(vision_model.patch_embed, 'patch_size'):
             patch_size = vision_model.patch_embed.patch_size
    elif hasattr(vision_model, 'conv1'):
        # openai clip style
        patch_size = vision_model.conv1.kernel_size
    
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
        
    if patch_size is None:
        return

    # Calculate new grid size
    new_grid_size = image_size // patch_size
    
    pos_embed = vision_model.positional_embedding
    if pos_embed.dim() != 2:
        return 
        
    num_tokens = pos_embed.shape[0]
    embed_dim = pos_embed.shape[1]
    
    # Assuming 1 class token
    num_patches = num_tokens - 1
    old_grid_size = int(math.sqrt(num_patches))
    
    if old_grid_size == new_grid_size:
        return

    print(f"Resizing positional embedding from {old_grid_size}x{old_grid_size} to {new_grid_size}x{new_grid_size}")
    
    cls_embed = pos_embed[0:1]
    patch_embed = pos_embed[1:]
    
    patch_embed = patch_embed.reshape(1, old_grid_size, old_grid_size, embed_dim).permute(0, 3, 1, 2)
    patch_embed = F.interpolate(patch_embed, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(-1, embed_dim)
    
    new_pos_embed = torch.cat((cls_embed, patch_embed), dim=0)
    vision_model.positional_embedding = torch.nn.Parameter(new_pos_embed)
