import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: MVTec 3D-AD Dataset'''
MVTEC3D_CLS_NAMES = [
    'bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
    'foam', 'peach', 'potato', 'rope', 'tire'
]

MVTEC3D_ROOT = os.path.join(DATA_ROOT, 'mvtec3d')

class MVTec3DDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=MVTEC3D_CLS_NAMES, aug_rate=0.0, root=MVTEC3D_ROOT, training=True):
        super(MVTec3DDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
