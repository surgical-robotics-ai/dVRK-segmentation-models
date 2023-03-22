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


class ImageSegmentationDataset(Dataset):

    img_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet normalize
        ]
    )

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

        image = Image.open(self.images_path_list[idx])
        annotation = Image.open(self.annotation_dir / self.images_path_list[idx].name)

        if transform:
            image = self.img_transforms(image)

        return {"img": image, "annotation": annotation}


if __name__ == "__main__":
    data_dir = Path("/home/juan1995/research_juan/accelnet_grant/data/rec01")

    ds = ImageSegmentationDataset(data_dir, "5colors")

    print(ds.images_id_list[1:15:2])
    print(ds.images_path_list[1:15:2])

    data = ds.__getitem__(100, transform=False)

    blended = blend_images(
        np.array(data["img"]).transpose(2, 0, 1),
        np.array(data["annotation"]).transpose(2, 0, 1)[:1, :, :],
        cmap="viridis",
        alpha=0.8,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(np.array(data["img"]))
    ax[1].imshow(np.array(data["annotation"]))
    ax[2].imshow(blended.transpose(1, 2, 0))
    [a.set_axis_off() for a in ax.squeeze()]
    fig.set_tight_layout(True)
    plt.show()

    pass
