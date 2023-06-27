from dataclasses import InitVar, dataclass, field
import json
from pathlib import Path
import re
from typing import List
import natsort
import numpy as np
import torch
from monai.bundle import ConfigParser
from monai.data import ThreadDataLoader
from surg_seg.Datasets.SegmentationLabelParser import (
    LabelInfoReader,
    SegmentationLabelInfo,
    SegmentationLabelParser,
)
from surg_seg.Datasets.ImageDataset import ImageDirParser, ImageSegmentationDataset
from surg_seg.Datasets.VideoDatasets import CombinedVidDataset
from surg_seg.ImageTransforms.ImageTransforms import ImageTransforms
from surg_seg.Networks.Models import create_FlexibleUnet
from surg_seg.Trainers.Trainer import ModelTrainer

##################################################################
# Concrete implementation of abstract classes
##################################################################


class Ambf5RecSegMapReader(LabelInfoReader):
    """Read the mapping file for ambf multi-class segmentation."""

    def __init__(self, mapping_file: Path, annotations_type: str):
        """
        Read segmentation labels mapping files

        Parameters
        ----------
        mapping_file : Path
        annotation_type : str
            Either [2colors, 4colors, or 5colors]
        """

        super().__init__(mapping_file)
        self.annotations_type = annotations_type

        self.read()

    def read(self):

        with open(self.mapping_file, "r") as f:
            mapper = json.load(f)

        if self.annotations_type in mapper:
            mask = mapper[self.annotations_type]
        else:
            raise RuntimeWarning(
                f"annotations type {self.annotations_type} not found in {self.path2mapping}"
            )

        self.classes_info = [
            SegmentationLabelInfo(idx, key, value) for idx, (key, value) in enumerate(mask.items())
        ]


class Ambf5RecDataReader(ImageDirParser):
    def __init__(self, root_dirs: List[Path], annotation_type: str):
        """Image dataset

        Parameters
        ----------
        root_dir : Path
        annotation_type : str
            Either [2colors, 4colors, or 5colors]
        """
        super().__init__(root_dirs)

        if not isinstance(root_dirs, list):
            root_dirs = [root_dirs]

        self.image_folder_list = []

        for root_dir in root_dirs:
            single_folder = SingleFolderReader(root_dir, annotation_type)
            self.image_folder_list.append(single_folder)
            self.images_list += single_folder.images_path_list
            self.labels_list += single_folder.label_path_list

    def __len__(self):
        return len(self.images_list)


@dataclass
class SingleFolderReader:
    """
    Read a single folder of data from the Ambf5Rec dataset
    """

    root_dir: Path
    annotation_type: InitVar[str]
    annotation_path: Path = field(init=False)
    image_path_list: List[Path] = field(init=False)
    label_path_list: List[Path] = field(init=False)
    image_id_list: List[int] = field(init=False)
    # Auxiliary variables used to identify duplicated ids in image folder
    flag_list: List[int] = field(init=False)

    def __post_init__(self, annotation_type):

        self.annotation_dir = self.__get_annotation_dir(annotation_type)

        self.images_path_list = natsort.natsorted(list((self.root_dir / "raw").glob("*.png")))
        self.flag_list = np.zeros(len(self.images_path_list))
        self.images_id_list = self.compute_id_list()

        self.label_path_list = [self.annotation_dir / img.name for img in self.images_path_list]

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


##################################################################
# Auxiliary functions
##################################################################

##################################################################
# Main functions
##################################################################


def train_with_video_dataset():
    device = "cpu"

    vid_root = Path("/home/juan1995/research_juan/accelnet_grant/data/rec01/")
    vid_filepath = vid_root / "raw/rec01_seg_raw.avi"
    seg_filepath = vid_root / "annotation2colors/rec01_seg_annotation2colors.avi"
    ds = CombinedVidDataset(vid_filepath, seg_filepath)
    dl = ThreadDataLoader(ds, batch_size=4, num_workers=0, shuffle=True)

    pretrained_weigths_path = Path("./assets/weights/trained-weights.pt")
    model = create_FlexibleUnet(device, pretrained_weigths_path, ds.label_channels)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)

    trainer = ModelTrainer(device=device, max_epochs=2)
    model, training_stats = trainer.train_model(model, optimizer, dl)

    training_stats.plot_stats()

    model_path = "./assets/weights/myweights_video"
    torch.save(model.state_dict(), model_path)
    training_stats.to_pickle(model_path)


def train_with_image_dataset():
    # Config parameters
    config = ConfigParser()
    config.read_config("./training_configs/thin7/ambf_train_config.yaml")
    train_config = config.get_parsed_content("ambf_train_config")

    train_dir_list = train_config["train_dir_list"]
    valid_dir_list = train_config["val_dir_list"]
    annotations_type = train_config["annotations_type"]
    pretrained_weights_path = train_config["pretrained_weights_path"]
    training_output_path = train_config["training_output_path"]
    mapping_file = train_config["mapping_file"]

    device = train_config["device"]
    epochs = train_config["epochs"]
    learning_rate = train_config["learning_rate"]

    # Train model
    train_data_reader = Ambf5RecDataReader(train_dir_list, annotations_type)
    valid_data_reader = Ambf5RecDataReader(valid_dir_list, annotations_type)
    label_info_reader = Ambf5RecSegMapReader(mapping_file, annotations_type)
    label_parser = SegmentationLabelParser(label_info_reader)

    ds = ImageSegmentationDataset(
        label_parser, train_data_reader, color_transforms=ImageTransforms.img_transforms
    )
    dl = ThreadDataLoader(ds, batch_size=4, num_workers=0, shuffle=True)

    val_ds = ImageSegmentationDataset(
        label_parser, valid_data_reader, color_transforms=ImageTransforms.img_transforms
    )
    val_dl = ThreadDataLoader(val_ds, batch_size=4, num_workers=0, shuffle=True)

    print(f"Training dataset size: {len(ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    model = create_FlexibleUnet(device, pretrained_weights_path, label_parser.mask_num)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    trainer = ModelTrainer(device=device, max_epochs=epochs)
    model, training_stats = trainer.train_model(model, optimizer, dl, validation_dl=val_dl)

    training_output_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), training_output_path / "myweights.pt")
    training_stats.to_pickle(training_output_path)
    training_stats.plot_stats(file_path=training_output_path)

    print(f"Last train IOU {training_stats.iou_list[-1]}")
    print(f"Last validation IOU {training_stats.validation_iou_list[-1]}")


def main():
    # train_with_video_dataset()
    train_with_image_dataset()


if __name__ == "__main__":
    main()
