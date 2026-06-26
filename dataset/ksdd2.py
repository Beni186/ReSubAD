import os

from .base_dataset import BaseDataset
from config import DATA_ROOT

"""Dataset source: KolektorSDD2."""
KSDD2_CLS_NAMES = [
    "Metal",
]
KSDD2_ROOT = os.path.join(DATA_ROOT, "ksdd2")


class KSDD2Dataset(BaseDataset):
    def __init__(
        self,
        transform,
        target_transform,
        clsnames=KSDD2_CLS_NAMES,
        aug_rate=0.0,
        root=KSDD2_ROOT,
        training=True,
        split=None,
    ):
        super(KSDD2Dataset, self).__init__(
            clsnames=clsnames,
            transform=transform,
            target_transform=target_transform,
            root=root,
            aug_rate=aug_rate,
            training=training,
            split=split
        )
