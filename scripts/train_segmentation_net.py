from pathlib import Path
from tqdm import trange
import torch

import monai
from monai.data import ThreadDataLoader
from monai.networks.nets import FlexibleUNet

from surg_seg.Datasets.ImageDataset import ImageSegmentationDataset
from surg_seg.Datasets.VideoDatasets import CombinedVidDataset
from surg_seg.Trainers.Trainer import ModelTrainer


def create_FlexibleUnet(device, pretrained_weights_path: Path, out_channels: int):

    model = FlexibleUNet(
        in_channels=3,
        out_channels=out_channels,
        backbone="efficientnet-b0",
        pretrained=True,
        is_pad=False,
    ).to(device)

    pretrained_weights = monai.bundle.load(
        name="endoscopic_tool_segmentation", bundle_dir=pretrained_weights_path, version="0.2.0"
    )
    model_weight = model.state_dict()
    weights_no_head = {k: v for k, v in pretrained_weights.items() if not "segmentation_head" in k}
    model_weight.update(weights_no_head)
    model.load_state_dict(model_weight)

    return model


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
    device = "cpu"
    root = Path("/home/juan1995/research_juan/accelnet_grant/data")
    train_dirs = [root / "rec01", root / "rec03", root / "rec05"]
    val_dirs = [root / "rec02", root / "rec04"]

    ds = ImageSegmentationDataset(train_dirs, "5colors")
    dl = ThreadDataLoader(ds, batch_size=4, num_workers=0, shuffle=True)

    val_ds = ImageSegmentationDataset(val_dirs, "5colors")
    val_dl = ThreadDataLoader(val_ds, batch_size=4, num_workers=0, shuffle=True)

    print(f"Training dataset size: {len(ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    pretrained_weigths_path = Path("./assets/weights/pretrained-weights")
    model = create_FlexibleUnet(device, pretrained_weigths_path, ds.label_channels)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)

    trainer = ModelTrainer(device=device, max_epochs=2)
    model, training_stats = trainer.train_model(model, optimizer, dl, validation_dl=val_dl)

    model_path = Path("./assets/weights/myweights_image")
    model_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path / "myweights.pt")
    training_stats.to_pickle(model_path)
    training_stats.plot_stats(file_path=model_path)

    print(f"Last train IOU {training_stats.iou_list[-1]}")
    print(f"Last validation IOU {training_stats.validation_iou_list[-1]}")


def main():
    # train_with_video_dataset()
    train_with_image_dataset()


if __name__ == "__main__":
    main()
