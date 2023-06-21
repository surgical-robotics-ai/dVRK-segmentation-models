from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import json
import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class SegmentationLabelInfo:
    """Helper class that relates current rgb color, name and id."""

    id: int
    name: str
    rgb: List[int]


@dataclass
class SegmentationLabelParser:
    """Class to convert between different segmentation annotations formats. Three formats are supported:
    1. rgb annotation.
    2. one-hot encoding.
    3. single channel encoding: single channel image where each pixel value is the class id.
    """

    label_info_reader: LabelInfoReader

    def __post_init__(self) -> None:

        self.__classes_info = self.label_info_reader.classes_info
        self.mask_num = len(self.__classes_info)

        if self.mask_num == 0:
            raise RuntimeError("No classes found. Maybe label_info_reader.read() was not called?")

    def get_classes_info(self) -> List[SegmentationLabelInfo]:
        return self.__classes_info

    def convert_rgb_to_single_channel(self, label_im, color_first=True):
        """Convert an annotations RGB image into a single channel image. The
        label image should have a shape `HWC` where `c==3`. This function
        converts labels that are compatible with Monai.blend function.

        Func to convert indexes taken from
        https://stackoverflow.com/questions/12138339/finding-the-x-y-indexes-of-specific-r-g-b-color-values-from-images-stored-in

        """

        assert label_im.shape[2] == 3, "label in wrong format"

        converted_img = np.zeros((label_im.shape[0], label_im.shape[1]))

        e: SegmentationLabelInfo
        for e in self.__classes_info:
            rgb = e.rgb
            new_color = e.id
            indices = np.where(np.all(label_im == rgb, axis=-1))
            converted_img[indices[0], indices[1]] = new_color

        converted_img = (
            np.expand_dims(converted_img, 0) if color_first else np.expand_dims(converted_img, -1)
        )
        return converted_img

    def convert_rgb_to_onehot(self, mask: np.ndarray):
        """Convert rgb label to one-hot encoding"""

        assert len(mask.shape) == 3, "label not a rgb image"
        assert mask.shape[2] == 3, "label not a rgb image"

        h, w, c = mask.shape

        ## Convert grey-scale label to one-hot encoding
        new_mask = np.zeros((self.mask_num, h, w))

        e: SegmentationLabelInfo
        for e in self.__classes_info:
            rgb = e.rgb
            new_idx = e.id
            new_mask[new_idx, :, :] = np.all(mask == rgb, axis=-1)

        return new_mask

    def convert_onehot_to_single_ch(self, onehot_mask: torch.tensor):
        m_temp = torch.argmax(onehot_mask, axis=0)
        m_temp = torch.unsqueeze(m_temp, 0)
        return m_temp

    def convert_single_ch_to_rgb(self, onehot_mask):

        shape = (onehot_mask.shape[0], onehot_mask.shape[1], 3)
        label_img = np.zeros(shape, dtype=np.uint8)

        e: SegmentationLabelInfo
        for e in self.__classes_info:
            rgb = e.rgb
            class_id = e.id
            label_img[onehot_mask == class_id] = rgb

        return label_img

    # new_mask = np.zeros((1, onehot_mask.shape[1], onehot_mask.shape[2]))
    #     e: LabelParserElement
    #     for e in self.conversion_list:

    #         temp = ((m_temp == e.id) * mask_value[idx]).data.numpy()
    #         new_mask += temp
    #     new_mask = np.expand_dims(new_mask, axis=-1)
    #     new_mask = np.concatenate((new_mask, new_mask, new_mask), axis=-1)
    #     new_mask = new_mask.astype(np.int32)
    #     return new_mask


class LabelInfoReader(ABC):
    """Abstract class to read segmentation labels metadata."""

    def __init__(self, mapping_file: Path):
        self.mapping_file: Path = mapping_file
        self.classes_info: List[SegmentationLabelInfo] = []

    @abstractmethod
    def read(self):
        """Read file and construct the classes_info list."""
        pass


class YamlSegMapReader(LabelInfoReader):
    """Read yaml mapping file for segmentation labels.

    Yaml files are formatted as ambf description files (ADF). First a unique list of `object_names` is defined.
    Afterwards, each object is defined by a `class_id` and a `rgb` color.
    Background should always be the first object to be defined

    Example file:
    -------------
    ```
    object_names:
        - background
        - needle
        - instrument

    background:
        class_id: 0
        rgb: [0, 0, 0]
    needle:
        class_id: 1
        rgb: [255, 0, 0]
    instrument:
        class_id: 1
        rgb: [0, 255, 0]
    ```
    """

    def __init__(self, mapping_file: Path):
        super().__init__(mapping_file)
        self.read()

    def read(self):

        with open(self.mapping_file, "r") as f:
            mapper = yaml.load(f, Loader=yaml.FullLoader)

            for object_name in mapper["object_names"]:
                new_seg_info = SegmentationLabelInfo(
                    mapper[object_name]["class_id"], object_name, mapper[object_name]["rgb"]
                )
                self.classes_info.append(new_seg_info)
