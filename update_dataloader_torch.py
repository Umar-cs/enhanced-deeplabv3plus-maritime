import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json


class LaRSDataset(Dataset):
    def __init__(
        self,
        image_list_path,
        image_dir,
        mask_dir,
        annotation_json_path,
        img_size,
        transform=None,
        debug: bool = False,
    ):
        """
        image_list_path: path to image_list.txt
        image_dir: folder with RGB images
        mask_dir: folder with semantic mask PNGs (or None for test)
        annotation_json_path: JSON with extra labels/metadata
        img_size: (W, H) tuple for resizing
        transform: optional Albumentations transform (applied to both image & mask)
        debug: if True, prints a bit of info; if False, stays quiet
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform
        self.debug = debug

        if self.debug:
            print(f"[DEBUG] LaRSDataset: image_dir={image_dir}, mask_dir={mask_dir}")

        # Load annotation JSON 
        with open(annotation_json_path, "r") as f:
            self.annotations = json.load(f)

        # Load image names from list
        with open(image_list_path, "r") as f:
            raw_names = [line.strip() for line in f if line.strip()]
        self.image_list = [n if n.lower().endswith(".jpg") else n + ".jpg" for n in raw_names]

        # Filter 
        filtered = []
        for img_name in self.image_list:
            img_path = os.path.join(self.image_dir, img_name)
            mask_name = os.path.splitext(img_name)[0] + ".png"
            mask_path = os.path.join(self.mask_dir, mask_name) if self.mask_dir else None

            if os.path.isfile(img_path) and (self.mask_dir is None or os.path.isfile(mask_path)):
                filtered.append(img_name)
            else:
                if self.debug:
                    print(f"[DEBUG] Skipping {img_name} - image or mask not found.")

        self.image_list = filtered

        if self.debug:
            print(f"[DEBUG] Loaded {len(self.annotations)} annotations.")
            print(f"[DEBUG] Using {len(self.image_list)} images with valid masks.")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.img_size, resample=Image.BILINEAR)

        mask = None
        if self.mask_dir is not None:
            mask_name = os.path.splitext(img_name)[0] + ".png"
            mask_path = os.path.join(self.mask_dir, mask_name)
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")

            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(self.img_size, resample=Image.NEAREST)

        # Extra labels from JSON
        annotation_info = self.annotations.get(img_name, {})
        labels = annotation_info.get("labels", [])

        # Albumentations transform
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=np.array(image), mask=np.array(mask))
                image = Image.fromarray(augmented["image"])
                mask = Image.fromarray(augmented["mask"])
            else:
                augmented = self.transform(image=np.array(image))
                image = Image.fromarray(augmented["image"])

        # To tensors
        image = transforms.ToTensor()(image)  # (C, H, W), [0, 1]
        if mask is not None:
            mask = torch.from_numpy(np.array(mask)).long()
        else:
            mask = torch.tensor([], dtype=torch.long)  # for test split

        if self.debug:
            print(f"[DEBUG] {img_name}: image min/max {image.min().item():.3f}/{image.max().item():.3f}")
            if mask.numel() > 0:
                print(f"[DEBUG] mask unique: {mask.unique()}")

        return image, mask, labels
