import os

from .base_dataset import BaseDataset


class PairedMaskDataset(BaseDataset):
    def __init__(
        self,
        clsnames,
        transform,
        target_transform,
        root,
        aug_rate=0.0,
        training=True,
        image_dir="images",
        mask_dir="masks",
        image_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
        mask_exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    ):
        meta_path = os.path.join(root, "meta.json")
        if os.path.isfile(meta_path):
            super(PairedMaskDataset, self).__init__(
                clsnames=clsnames,
                transform=transform,
                target_transform=target_transform,
                root=root,
                aug_rate=aug_rate,
                training=training,
            )
            return

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.aug_rate = aug_rate
        self.training = training
        self.cls_names = clsnames
        self.meta_info = {cls_name: [] for cls_name in self.cls_names}

        for cls_name in self.cls_names:
            self.meta_info[cls_name] = self._build_paired_info(
                cls_name=cls_name,
                image_dir=image_dir,
                mask_dir=mask_dir,
                image_exts=image_exts,
                mask_exts=mask_exts,
            )

        self.data_all = []
        for cls_name in self.cls_names:
            self.data_all.extend(self.meta_info[cls_name])
        self.length = len(self.data_all)

    def _build_paired_info(self, cls_name, image_dir, mask_dir, image_exts, mask_exts):
        image_root = os.path.join(self.root, image_dir)
        mask_root = os.path.join(self.root, mask_dir)
        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"Image directory not found: {image_root}")
        if not os.path.isdir(mask_root):
            raise FileNotFoundError(f"Mask directory not found: {mask_root}")

        mask_by_stem = {}
        for mask_name in sorted(os.listdir(mask_root)):
            stem, ext = os.path.splitext(mask_name)
            if ext.lower() in mask_exts:
                mask_by_stem[stem] = os.path.join(mask_dir, mask_name)

        cls_info = []
        for img_name in sorted(os.listdir(image_root)):
            stem, ext = os.path.splitext(img_name)
            if ext.lower() not in image_exts:
                continue
            if stem not in mask_by_stem:
                raise FileNotFoundError(f"Mask not found for image: {os.path.join(image_root, img_name)}")
            cls_info.append(
                {
                    "img_path": os.path.join(image_dir, img_name),
                    "mask_path": mask_by_stem[stem],
                    "cls_name": cls_name,
                    "specie_name": "defect",
                    "anomaly": 1,
                }
            )

        if len(cls_info) == 0:
            raise RuntimeError(f"No paired samples found in {image_root} and {mask_root}")
        return cls_info
