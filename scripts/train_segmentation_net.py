from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from tqdm import trange
import torch

import monai
from monai.data import ThreadDataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import FlexibleUNet
import monai.transforms as mt

from dataclasses import dataclass, field
from surg_seg.Datasets.ImageDataset import ImageSegmentationDataset
from surg_seg.Datasets.VideoDatasets import CombinedVidDataset
import pickle


@dataclass
class TrainingStats:
    loss_list: List[float] = field(default_factory=list)
    iou_list: List[float] = field(default_factory=list)
    epoch_list: List[float] = field(default_factory=list)

    def add_element(self, epoch, loss, iou):
        self.epoch_list.append(epoch)
        self.loss_list.append(loss)
        self.iou_list.append(iou)

    def plot_stats(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor="white")
        ax.set_xlabel("Epoch")
        ax.plot(self.epoch_list, self.loss_list, label="loss")
        ax.plot(self.epoch_list, self.iou_list, label="iou")
        plt.legend()
        plt.show()

    def to_pickle(self, path: Path):
        pickle.dump(self, open(path / "training_stats.pkl", "wb"))


@dataclass
class ModelTrainer:
    device: str = "cuda"
    max_epochs: int = 20
    val_interval: int = 1
    best_metric: int = -1
    best_metric_epoch: int = -1

    def __post_init__(self):
        self.dice_metric = DiceMetric(reduction="mean")
        self.iou_metric = MeanIoU(reduction="mean")
        self.post_trans = mt.Compose(
            [
                mt.Activations(sigmoid=True),
                mt.AsDiscrete(threshold=0.5),
            ]
        )

        self.loss_function = DiceLoss(sigmoid=True)

    def train_model(
        self, model: torch.nn.Module, optimizer: torch.optim.Adam, dl
    ) -> Tuple[torch.nn.Module, TrainingStats]:
        training_stats = TrainingStats()

        tr = trange(self.max_epochs)
        for epoch in tr:
            model.train()
            epoch_loss = 0

            for batch_data in dl:
                inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(
                    self.device
                )
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                pred = [self.post_trans(x) for x in decollate_batch(outputs)]
                label = [x for x in decollate_batch(labels)]
                batch_iou = self.iou_metric(y_pred=pred, y=label)

            epoch_iou = self.iou_metric.aggregate().item()
            self.iou_metric.reset()
            epoch_loss /= len(dl)
            training_stats.add_element(epoch, epoch_loss, epoch_iou)

            # Update progress bar
            tr.set_description(f"Loss: {epoch_loss:.4f}")

        return model, training_stats


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
    data_dir = Path("/home/juan1995/research_juan/accelnet_grant/data/rec03")
    ds = ImageSegmentationDataset(data_dir, "5colors")
    dl = ThreadDataLoader(ds, batch_size=4, num_workers=0, shuffle=True)

    pretrained_weigths_path = Path("./assets/weights/trained-weights")
    model = create_FlexibleUnet(device, pretrained_weigths_path, ds.label_channels)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)

    trainer = ModelTrainer(device=device, max_epochs=2)
    model, training_stats = trainer.train_model(model, optimizer, dl)

    training_stats.plot_stats()
    model_path = "./assets/weights/myweights_image"
    torch.save(model.state_dict(), model_path)
    training_stats.to_pickle(model_path)


def main():
    # train_with_video_dataset()
    train_with_image_dataset()


if __name__ == "__main__":
    main()
