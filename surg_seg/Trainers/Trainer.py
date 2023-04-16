from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
from tqdm import trange
import torch

from monai.data import decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
import monai.transforms as mt

from dataclasses import dataclass, field
import pickle


@dataclass
class TrainingStats:
    loss_list: List[float] = field(default_factory=list)
    iou_list: List[float] = field(default_factory=list)
    validation_iou_list: List[float] = field(default_factory=list)
    epoch_list: List[float] = field(default_factory=list)

    def add_element(self, epoch, loss, iou):
        self.epoch_list.append(epoch)
        self.loss_list.append(loss)
        self.iou_list.append(iou)

    def plot_stats(self, file_path: Path = None):
        fig, ax = plt.subplots(1, 2, figsize=(6, 6), facecolor="white")
        [axi.set_xlabel("Epoch") for axi in ax.squeeze()]
        ax[0].plot(self.epoch_list, self.loss_list, label="loss")
        ax[1].plot(self.epoch_list, self.iou_list, label="train iou")
        ax[1].plot(self.epoch_list, self.validation_iou_list, label="val iou")

        if file_path is not None:
            plt.savefig(file_path / "training_stats.png")

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
        self.iou_metric = MeanIoU(include_background=False, reduction="mean")
        self.post_trans = mt.Compose(
            [
                mt.Activations(sigmoid=True),
                mt.AsDiscrete(threshold=0.5),
            ]
        )

        self.loss_function = DiceLoss(sigmoid=True)

    def train_model(
        self, model: torch.nn.Module, optimizer: torch.optim.Adam, training_dl, validation_dl=None
    ) -> Tuple[torch.nn.Module, TrainingStats]:
        training_stats = TrainingStats()

        tr = trange(self.max_epochs)
        for epoch in tr:
            model.train()
            epoch_loss = 0

            for batch_data in training_dl:
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
            epoch_loss /= len(training_dl)
            training_stats.add_element(epoch, epoch_loss, epoch_iou)

            # Update progress bar
            tr.set_description(f"Loss: {epoch_loss:.4f}")

            if validation_dl is not None:
                model.eval()
                validation_iou = self.calculate_validation_iou(model, validation_dl)
                training_stats.validation_iou_list.append(validation_iou)

        return model, training_stats

    def calculate_validation_iou(self, model, validation_dl):
        with torch.no_grad():
            for batch_data in validation_dl:
                inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(
                    self.device
                )
                outputs = model(inputs)

                pred = [self.post_trans(x) for x in decollate_batch(outputs)]
                label = [x for x in decollate_batch(labels)]
                batch_iou = self.iou_metric(y_pred=pred, y=label)

        validation_iou = self.iou_metric.aggregate().item()
        self.iou_metric.reset()
        return validation_iou
