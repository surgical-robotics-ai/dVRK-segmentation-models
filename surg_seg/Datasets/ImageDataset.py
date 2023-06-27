from __future__ import annotations
from abc import ABC
import json
from pathlib import Path
from typing import Callable, List
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
import re
import natsort

from monai.visualize.utils import blend_images
from dataclasses import InitVar, dataclass, field
from surg_seg.Datasets.SegmentationLabelParser import LabelInfoReader, SegmentationLabelParser
from surg_seg.ImageTransforms.ImageTransforms import ImageTransforms


@dataclass
class ImageSegmentationDataset(Dataset):
    label_parser: SegmentationLabelParser
    img_dir_parser: ImageDirParser
    color_transforms: T.Compose = None
    geometric_transforms: Callable = None

    def __post_init__(self):
        self.images_list: List[Path] = self.img_dir_parser.images_list
        self.labels_list: List[Path] = self.img_dir_parser.labels_list

        if self.color_transforms is None:
            self.color_transforms = ImageTransforms.img_transforms

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx, transform: bool = True):
        if isinstance(idx, slice):
            RuntimeError("Slices are not supported")

        image = np.array(Image.open(self.images_list[idx]))
        annotation = np.array(Image.open(self.labels_list[idx]))

        if transform:
            image = self.color_transforms(image)
            annotation = self.label_parser.convert_rgb_to_onehot(annotation)
            annotation = torch.tensor(annotation)

            if self.geometric_transforms is not None:
                image, annotation = self.geometric_transforms(image, annotation)

        return {"image": image, "label": annotation}


class ImageDirParser(ABC):
    def __init__(self, root_dirs: List[Path]):
        self.root_dirs: List[Path] = root_dirs
        self.images_list: List[Path] = []
        self.labels_list: List[Path] = []

    def __len__(self):
        return len(self.images_list)


def display_transformed_images(idx: int, ds: ImageSegmentationDataset):

    data = ds[idx]  # Get transformed images
    # print(f"one-hot shape: {data['label'].shape}")

    single_ch_annotation = ds.label_parser.convert_onehot_to_single_ch(data["label"])
    single_ch_annotation = np.array(single_ch_annotation)
    # single_ch_annotation = ds.label_parser.convert_rgb_to_single_channel(raw_label)
    raw_image = np.array(ImageTransforms.inv_transforms(data["image"]))
    raw_label = ds.__getitem__(idx, transform=False)["label"]

    blended = blend_images(
        raw_image,
        single_ch_annotation,
        cmap="viridis",
        alpha=0.8,
    )

    display_images(raw_image.transpose(1, 2, 0), raw_label, blended.transpose(1, 2, 0))


def display_untransformed_images(idx: int, ds: ImageSegmentationDataset):
    data = ds.__getitem__(idx, transform=False)  # Get raw images

    raw_image = data["image"]
    raw_label = data["label"]
    onehot = ds.label_parser.convert_rgb_to_onehot(raw_label)

    # fake_annotation = np.zeros_like(np.array(data["image"]))
    # fake_annotation[:40, :40] = [1, 1, 1]
    # fake_annotation[40:80, 40:80] = [2, 2, 2]
    # fake_annotation[80:120, 80:120] = [3, 3, 3]

    single_ch_label = ds.label_parser.convert_rgb_to_single_channel(raw_label)
    blended = blend_images(
        raw_image.transpose(2, 0, 1),
        single_ch_label,
        cmap="viridis",
        alpha=0.8,
    )

    display_images(raw_image, raw_label, blended.transpose(1, 2, 0))


def display_images(img, label, blended):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(label)
    ax[2].imshow(blended)
    [a.set_axis_off() for a in ax.squeeze()]
    fig.set_tight_layout(True)
    plt.show()


if __name__ == "__main__":
    root = Path("/home/juan1995/research_juan/accelnet_grant/data")
    data_dirs = [root / "rec01", root / "rec02", root / "rec03", root / "rec04", root / "rec05"]

    ds = ImageSegmentationDataset(data_dirs, "5colors")

    print(f"length of dataset: {len(ds)}")

    display_untransformed_images(100, ds)
    display_transformed_images(100, ds)
    display_transformed_images(230, ds)
    display_transformed_images(330, ds)
    display_transformed_images(430, ds)
    display_transformed_images(530, ds)
