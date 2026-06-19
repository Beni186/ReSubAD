"""
Shared configuration for the exp_ca zero-shot anomaly experiments.
"""
import os

# Root directory where processed anomaly datasets live. Can be overridden via env.
DATA_ROOT = os.getenv("EXP_CA_DATA_ROOT", "../datasets")

# Default CLIP backbone and weights.
DEFAULT_CLIP_MODEL = os.getenv("EXP_CA_CLIP_MODEL", "ViT-L/14@336px")
DEFAULT_CLIP_PRETRAINED = os.getenv(
    "EXP_CA_CLIP_PRETRAINED", os.path.join(os.path.dirname(__file__), "model", "ViT-L-14-336px.pt")
)

# Image size used for both CLIP preprocessing and mask resizing.
DEFAULT_IMAGE_SIZE = int(os.getenv("EXP_CA_IMAGE_SIZE", "518"))

# Training / evaluation defaults.
DEFAULT_BATCH_SIZE = int(os.getenv("EXP_CA_BATCH_SIZE", "4"))
DEFAULT_DATASET = os.getenv("EXP_CA_DATASET", "mvtec")

# Temperature for scaling cosine similarities before softmax.
DEFAULT_TEMPERATURE = float(os.getenv("EXP_CA_TEMPERATURE", "0.07"))
