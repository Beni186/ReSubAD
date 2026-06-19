import os

from .paired_mask import PairedMaskDataset
from config import DATA_ROOT

"""Dataset source: https://datasets.simula.no/kvasir-seg/"""
KVASIR_CLS_NAMES = [
    "Kvasir",
]
KVASIR_ROOT = os.path.join(DATA_ROOT, "kvasir")


class KvasirDataset(PairedMaskDataset):
    def __init__(
        self,
        transform,
        target_transform,
        clsnames=KVASIR_CLS_NAMES,
        aug_rate=0.0,
        root=KVASIR_ROOT,
        training=True,
    ):
        super(KvasirDataset, self).__init__(
            clsnames=clsnames,
            transform=transform,
            target_transform=target_transform,
            root=root,
            aug_rate=aug_rate,
            training=training,
        )
