import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: Real3D-AD Dataset'''
REAL3D_CLS_NAMES = [
    'airplane', 'candybar', 'car', 'chicken', 'diamond', 'duck', 
    'fish', 'gemstone', 'seahorse', 'shell', 'starfish', 'toffees'
]

REAL3D_ROOT = os.path.join(DATA_ROOT, 'real3d')

class Real3DDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=REAL3D_CLS_NAMES, aug_rate=0.0, root=REAL3D_ROOT, training=True, split=None):
        super(Real3DDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training,
            split=split
        )