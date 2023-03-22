import json
from pathlib import Path
from typing import List
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
import re
import natsort

from monai.visualize.utils import blend_images
from dataclasses import dataclass


@dataclass
class LabelParser:
    path2mapping: Path
    annotations_type: str

    def __post_init__(self):

        with open(self.path2mapping, "r") as f:
            self.mapper = json.load(f)
        self.mask = self.mapper[self.annotations_type]

        self.mask_num = len(self.mask)
        self.conversion_list = [
            (key, idx, value) for idx, (key, value) in enumerate(self.mask.items())
        ]

    def convert_rgb_to_single_channel(self, label_im, color_first=True):
        """Convert an annotations RGB image into a single channel image. The
        label image should have a shape `HWC` where `c==3`. This function
        converts labels that are compatible with Monai.blend function.

        Func to convert indexes taken from
        https://stackoverflow.com/questions/12138339/finding-the-x-y-indexes-of-specific-r-g-b-color-values-from-images-stored-in

        """

        assert label_im.shape[2] == 3, "label in wrong format"

        converted_img = np.zeros((label_im.shape[0], label_im.shape[1]))
        for e in self.conversion_list:
            rgb = e[2]
            new_color = e[1]
            indices = np.where(np.all(label_im == rgb, axis=-1))
            converted_img[indices[0], indices[1]] = new_color

        converted_img = (
            np.expand_dims(converted_img, 0) if color_first else np.expand_dims(converted_img, -1)
        )
        return converted_img


class Transforms:

    img_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet normalize
        ]
    )
    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    inv_transforms = T.Compose(
        [
            T.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )


class ImageSegmentationDataset(Dataset):
    def __init__(self, root_dir: Path, annotation_type: str):
        """Image dataset

        Parameters
        ----------
        root_dir : Path
        annotation_type : str
            Either [2colors, 4colors, or 5colors]
        """
        self.root_dir = root_dir
        self.annotation_dir = self.__get_annotation_dir(annotation_type)

        self.images_path_list = natsort.natsorted(list((self.root_dir / "raw").glob("*.png")))
        self.flag_list = np.zeros(len(self.images_path_list))
        self.images_id_list = self.compute_id_list()

        self.label_parser = LabelParser(self.root_dir / "mapping.json", annotation_type)

    def compute_id_list(self):
        ids = []
        img_name: Path
        for img_name in self.images_path_list:
            id_match = self.__extract_id(img_name.name)
            self.__check_and_mark_id(id_match)
            ids.append(id_match)
        return ids

    def __get_annotation_dir(self, annotation_type):
        valid_options = ["2colors", "4colors", "5colors"]
        if annotation_type not in valid_options:
            raise RuntimeError(
                f"{annotation_type} is not a valid annotation.\n Valid annotations are {valid_options}"
            )
        return self.root_dir / ("annotation" + annotation_type)

    def __extract_id(self, img_name: str) -> int:
        """Extract id from image name"""
        id_match = re.findall("[0-9]{6}", img_name)

        if len(id_match) == 0:
            raise RuntimeError(f"Image {img_name} not formatted correctly")

        id_match = int(id_match[0])
        return id_match

    def __check_and_mark_id(self, id_match):
        """Check that there are no duplicated id"""
        if self.flag_list[id_match]:
            raise RuntimeError(f"Id {id_match} is duplicated")

        self.flag_list[id_match] = 1

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, idx, transform=True):
        if isinstance(idx, slice):
            RuntimeError("Slices are not supported")

        image = np.array(Image.open(self.images_path_list[idx]))
        annotation = np.array(Image.open(self.annotation_dir / self.images_path_list[idx].name))

        if transform:
            image = Transforms.img_transforms(image)

        return {"img": image, "annotation": annotation}


def display_transformed_images(idx: int, ds: ImageSegmentationDataset):

    data = ds[idx]  # Get transformed images
    raw_label = data["annotation"]
    single_ch_annotation = ds.label_parser.convert_rgb_to_single_channel(raw_label)
    raw_image = np.array(Transforms.inv_transforms(data["img"]))

    blended = blend_images(
        raw_image,
        single_ch_annotation,
        cmap="viridis",
        alpha=0.8,
    )

    display_images(raw_image.transpose(1, 2, 0), raw_label, blended.transpose(1, 2, 0))


def display_untransformed_images(idx: int, ds: ImageSegmentationDataset):
    data = ds.__getitem__(100, transform=False)  # Get raw images

    raw_image = data["img"]
    raw_label = data["annotation"]

    fake_annotation = np.zeros_like(np.array(data["img"]))
    fake_annotation[:40, :40] = [1, 1, 1]
    fake_annotation[40:80, 40:80] = [2, 2, 2]
    fake_annotation[80:120, 80:120] = [3, 3, 3]

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
    data_dir = Path("/home/juan1995/research_juan/accelnet_grant/data/rec01")

    ds = ImageSegmentationDataset(data_dir, "4colors")

    display_untransformed_images(100, ds)
    display_transformed_images(140, ds)
