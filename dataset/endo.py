import os

from .paired_mask import PairedMaskDataset
from config import DATA_ROOT

"""Dataset source: EndoTect 2020 Segmentation Test Dataset."""
ENDO_CLS_NAMES = [
    "Endo",
]
ENDO_ROOT = os.path.join(DATA_ROOT, "endo")


class EndoDataset(PairedMaskDataset):
    def __init__(
        self,
        transform,
        target_transform,
        clsnames=ENDO_CLS_NAMES,
        aug_rate=0.0,
        root=ENDO_ROOT,
        training=True,
    ):
        super(EndoDataset, self).__init__(
            clsnames=clsnames,
            transform=transform,
            target_transform=target_transform,
            root=root,
            aug_rate=aug_rate,
            training=training,
        )
