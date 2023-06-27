from pathlib import Path
from natsort import natsorted
import numpy as np
from typing import List, Tuple
from matplotlib import pyplot as plt
from tqdm import tqdm
from monai.visualize.utils import blend_images
import torch
from monai.bundle import ConfigParser
from monai.data import ThreadDataLoader

from surg_seg.Datasets.SegmentationLabelParser import (
    SegmentationLabelParser,
    YamlSegMapReader,
)
from surg_seg.Datasets.ImageDataset import (
    ImageDirParser,
    ImageSegmentationDataset,
)
from surg_seg.ImageTransforms.ImageTransforms import ImageTransforms
from surg_seg.Metrics.MetricsUtils import AggregatedMetricTable, IOUStats
from surg_seg.Networks.Models import FlexibleUnet1InferencePipe, create_FlexibleUnet
from surg_seg.Trainers.Trainer import ModelTrainer

##################################################################
# Concrete implementation of abstract classes
##################################################################
class CustomImageDirParser(ImageDirParser):
    def __init__(self, root_dirs: List[Path]):
        super().__init__(root_dirs)

        self.parse_image_dir(root_dirs[0])

    def parse_image_dir(self, root_dir: Path):
        self.images_list = natsorted(list((root_dir / "raw").glob("*.png")))
        self.labels_list = natsorted(list((root_dir / "label").glob("*.png")))


##################################################################
# Auxiliary functions
##################################################################
def create_label_parser(config: ConfigParser) -> SegmentationLabelParser:
    train_config = config.get_parsed_content("ambf_train_config")
    mapping_file = train_config["mapping_file"]
    label_info_reader = YamlSegMapReader(mapping_file)
    label_parser = SegmentationLabelParser(label_info_reader)

    return label_parser


def create_train_dataset_and_dataloader(
    config: ConfigParser, label_parser: SegmentationLabelParser, batch_size: int
) -> Tuple[ImageSegmentationDataset, ThreadDataLoader]:

    train_config = config.get_parsed_content("ambf_train_config")
    train_dir_list = train_config["train_dir_list"]
    train_data_reader = CustomImageDirParser(train_dir_list)

    ds = ImageSegmentationDataset(
        label_parser,
        train_data_reader,
        color_transforms=ImageTransforms.img_transforms,
        geometric_transforms=ImageTransforms.geometric_transforms,
    )
    dl = ThreadDataLoader(ds, batch_size=batch_size, num_workers=2, shuffle=True)

    return ds, dl


def create_valid_dataset_and_dataloader(
    config: ConfigParser, label_parser: SegmentationLabelParser, batch_size: int
) -> Tuple[ImageSegmentationDataset, ThreadDataLoader]:
    train_config = config.get_parsed_content("ambf_train_config")
    valid_dir_list = train_config["val_dir_list"]

    valid_data_reader = CustomImageDirParser(valid_dir_list)

    val_ds = ImageSegmentationDataset(
        label_parser, valid_data_reader, color_transforms=ImageTransforms.img_transforms
    )
    val_dl = ThreadDataLoader(val_ds, batch_size=batch_size, num_workers=2, shuffle=True)

    return val_ds, val_dl


##################################################################
# Main functions
##################################################################


def train_with_image_dataset(config: ConfigParser):
    train_config = config.get_parsed_content("ambf_train_config")
    device = train_config["device"]

    # Load data
    label_parser = create_label_parser(config)
    ds, dl = create_train_dataset_and_dataloader(config, label_parser, batch_size=8)
    val_ds, val_dl = create_valid_dataset_and_dataloader(config, label_parser, batch_size=8)

    print(f"Training dataset size: {len(ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Number of output clases: {label_parser.mask_num}")

    # Load model
    pretrained_weights_path = train_config["pretrained_weights_path"]
    model = create_FlexibleUnet(device, pretrained_weights_path, label_parser.mask_num)

    # Load trainer
    training_output_path = train_config["training_output_path"]
    epochs = train_config["epochs"]
    learning_rate = train_config["learning_rate"]

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    trainer = ModelTrainer(device=device, max_epochs=epochs)
    model, training_stats = trainer.train_model(model, optimizer, dl, validation_dl=val_dl)

    # Save model
    training_output_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), training_output_path / "myweights.pt")
    training_stats.to_pickle(training_output_path)
    training_stats.plot_stats(file_path=training_output_path)

    print(f"Last train IOU {training_stats.iou_list[-1]}")
    print(f"Last validation IOU {training_stats.validation_iou_list[-1]}")


def show_images(config: ConfigParser, show_valid: str = False):
    train_config = config.get_parsed_content("ambf_train_config")

    label_parser = create_label_parser(config)
    if show_valid:
        print("Showing validation images")
        ds, dl = create_valid_dataset_and_dataloader(config, label_parser, batch_size=1)
    else:
        print("Showing training images")
        ds, dl = create_valid_dataset_and_dataloader(config, label_parser, batch_size=1)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.set_tight_layout(True)
    fig.subplots_adjust(hspace=0, wspace=0)
    for i, ax in enumerate(axes.flat):
        pair = next(iter(dl))
        im = pair["image"][0]
        lb = pair["label"][0]

        im = ImageTransforms.inv_transforms(im)
        lb = label_parser.convert_onehot_to_single_ch(lb)
        blended = blend_images(im, lb, cmap="viridis", alpha=0.7)
        blended = blended.numpy().transpose(1, 2, 0)
        blended = (blended * 255).astype(np.uint8)
        ax.imshow(blended)
        ax.axis("off")

    plt.show()


def show_inference_samples(config: ConfigParser):
    device = "cuda"
    path2weights = config.get_parsed_content("test#weights")

    label_parser = create_label_parser(config)
    ds, dl = create_valid_dataset_and_dataloader(config, label_parser, batch_size=1)

    model_pipe = FlexibleUnet1InferencePipe(
        path2weights, device, out_channels=label_parser.mask_num
    )
    model_pipe.model.eval()

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.set_tight_layout(True)
    fig.subplots_adjust(hspace=0, wspace=0)
    for i, ax in enumerate(axes.flat):
        # pair = next(iter(dl))
        pair = ds.__getitem__(i, transform=False)
        im = pair["image"]
        lb = pair["label"]
        print(im.shape)
        input_tensor, inferred_single_ch = model_pipe.infer(im)

        inferred_single_ch = inferred_single_ch.detach().cpu()
        input_tensor = input_tensor.detach().cpu()[0]
        blended = blend_images(input_tensor, inferred_single_ch, cmap="viridis", alpha=0.8).numpy()
        blended = (np.transpose(blended, (1, 2, 0)) * 254).astype(np.uint8)

        # im = ImageTransforms.inv_transforms(im)
        # lb = label_parser.convert_onehot_to_single_ch(lb)
        # blended = blend_images(im, lb, cmap="viridis", alpha=0.7)
        # blended = blended.numpy().transpose(1, 2, 0)
        # blended = (blended * 255).astype(np.uint8)
        ax.imshow(blended)
        ax.axis("off")
    plt.show()


def calculate_metrics_on_valid(config: ConfigParser):
    device = "cuda"
    path2weights = config.get_parsed_content("test#weights")

    label_parser = create_label_parser(config)
    ds, dl = create_valid_dataset_and_dataloader(config, label_parser, batch_size=1)

    model_pipe = FlexibleUnet1InferencePipe(
        path2weights, device, out_channels=label_parser.mask_num
    )
    model_pipe.model.eval()

    iou_stats = IOUStats(label_parser)
    for batch in tqdm(dl, desc="Calculating metrics"):
        img = batch["image"].to(device)
        label = batch["label"]
        img_paths = ["empty"] * label.shape[0]

        prediction = model_pipe.model(img).detach().cpu()
        onehot_prediction = ImageTransforms.predictions_transforms(prediction)
        iou_stats.calculate_metrics_from_batch(onehot_prediction, label, img_paths)

        # img = img.detach().cpu()[0]
        # # img = ImageTransforms.inv_transforms(img).type(torch.uint8)[0].numpy()
        # single_ch_prediction = onehot_prediction[0].argmax(dim=0, keepdim=True)
        # blended = blend_images(img, single_ch_prediction, cmap="viridis", alpha=0.8).numpy()
        # blended = (np.transpose(blended, (1, 2, 0)) * 254).astype(np.uint8)
        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(blended)
        # # ax.imshow(np.transpose(img, (1, 2, 0)))
        # plt.show()

    iou_stats.calculate_aggregated_stats()
    table = AggregatedMetricTable(iou_stats)
    table.fill_table()
    table.print_table()


def main():
    # Config parameters
    config = ConfigParser()
    config.read_config("./training_configs/thin7/dvrk_train_config.yaml")

    # show_images(config, show_valid=True)

    train_with_image_dataset(config)

    # show_inference_samples(config)

    # calculate_metrics_on_valid(config)


if __name__ == "__main__":
    main()
