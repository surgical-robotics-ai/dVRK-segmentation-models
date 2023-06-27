from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from typing import List
from monai.metrics import compute_iou
from monai.metrics import DiceMetric, MeanIoU
import numpy as np

import torch
from surg_seg.Metrics.MkDownTableGen import MarkdownTable
from surg_seg.Datasets.SegmentationLabelParser import SegmentationLabelParser


@dataclass
class AggregatedStat:
    mean: float
    median: float
    std: float
    min: float
    max: float
    N: int
    metric_name: str = None
    class_name: str = None

    @classmethod
    def from_array(cls, array: np.ndarray, metric_name: str = None, class_name: str = None):
        N = np.count_nonzero(~np.isnan(array))

        return cls(
            mean=np.nanmean(array),
            median=np.nanmedian(array),
            std=np.nanstd(array),
            min=np.nanmin(array),
            max=np.nanmax(array),
            N=N,
            metric_name=metric_name,
            class_name=class_name,
        )


@dataclass
class SingleStat:
    image_name: str
    metric_value: float
    label_name: str
    metric_name: str

    def __str__(self):
        return f"{self.image_name}: {self.metric_value}"


class SingleStatList(list):
    def __init__(self, label_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_name = label_name

    def append(self, item: SingleStat):
        if item.label_name != self.label_name:
            raise ValueError(f"Stat list can only contain stats for label {self.label_name}")

        if not isinstance(item, SingleStat):
            raise TypeError("item must be of type SingleImageStat")
        else:
            super().append(item)

    def sort(self, reverse=False):
        super().sort(key=lambda x: x.metric_value, reverse=reverse)

    def get_all_values(self):
        return [item.metric_value for item in self]


class AggregatedStatPerLabel(ABC):
    """Abstract class for aggregate statistics."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aggregated_stats_dict: dict[str, AggregatedStat] = {}


class SingleStatsPerLabel(ABC):
    def __init__(
        self, label_parser: SegmentationLabelParser, other_labels: List[str], **kwargs
    ) -> None:

        super().__init__(**kwargs)
        self.label_parser = label_parser
        self.single_stats_dict: dict[str, SingleStatList] = {}

        for label_info in label_parser.get_classes_info():
            self.single_stats_dict[label_info.name] = SingleStatList(label_name=label_info.name)

        for other_label in other_labels:
            self.single_stats_dict[other_label] = SingleStatList(label_name=other_label)


class IOUStats(SingleStatsPerLabel, AggregatedStatPerLabel):
    def __init__(self, label_parser: SegmentationLabelParser) -> None:
        self.other_labels = ["all", "all_no_background"]
        super().__init__(label_parser=label_parser, other_labels=self.other_labels)
        self.label_parser = label_parser
        self.metric_name: str = "IOU"

    def calculate_metrics_from_batch(
        self, onehot_pred: torch.Tensor, onehot_labels: torch.Tensor, image_names: list[str]
    ):
        """Calculate metrics from a batch of predictions and labels. The function expect predictions as
        an one-hot encoded tensor with shape (`BNHW` where `N` corresponds to the number of classes).

        - epoch_iou shape -> (batch_size, num_classes)
        """

        assert onehot_pred.min() >= 0.0, "predictions must be one-hot encoded"
        assert onehot_pred.max() <= 1.0, "predictions must be one-hot encoded"

        batch_iou = compute_iou(onehot_pred, onehot_labels, include_background=True)

        assert (
            np.nanmin(batch_iou.numpy()) >= 0.0 and np.nanmax(batch_iou.numpy()) <= 1.0
        ), f"IOU must be between 0 and 1. currently: [{batch_iou.min()},{batch_iou.max()}]"

        for img_idx in range(batch_iou.shape[0]):
            for label_idx, label_info in enumerate(self.label_parser.get_classes_info()):
                label_name = label_info.name
                value = batch_iou[img_idx, label_idx].item()

                self.single_stats_dict[label_name].append(
                    SingleStat(image_names[img_idx], value, label_name, self.metric_name)
                )

            # Get mean IOU for all labels
            label_name = "all"
            value = batch_iou[img_idx, :].mean().item()
            self.single_stats_dict[label_name].append(
                SingleStat(image_names[img_idx], value, label_name, self.metric_name)
            )

            label_name = "all_no_background"
            value = batch_iou[img_idx, 1:].mean().item()
            self.single_stats_dict[label_name].append(
                SingleStat(image_names[img_idx], value, label_name, self.metric_name)
            )

    def calculate_aggregated_stats(self):
        for label_name, stat_list in self.single_stats_dict.items():
            all_values = np.array(stat_list.get_all_values())

            assert len(all_values.shape) == 1, "all_values must be a 1D array"

            aggregated = AggregatedStat.from_array(all_values, self.metric_name, label_name)
            self.aggregated_stats_dict[label_name] = aggregated


@dataclass
class AggregatedMetricTable:
    aggregated_stats: AggregatedStatPerLabel

    def __post_init__(self):
        self.table = MarkdownTable(headers=["label", "mean", "median", "std", "min", "max", "N"])
        self.aggregated_stats_dict = self.aggregated_stats.aggregated_stats_dict

    def fill_table(self):
        for label_name, aggregated_stat in self.aggregated_stats_dict.items():
            data_dict = {
                "label": label_name,
                "mean": aggregated_stat.mean,
                "median": aggregated_stat.median,
                "std": aggregated_stat.std,
                "min": aggregated_stat.min,
                "max": aggregated_stat.max,
                "N": aggregated_stat.N,
            }
            self.table.add_data(**data_dict)

    def print_table(self, floatfmt=".2f"):
        self.table.print(floatfmt=floatfmt)


if __name__ == "__main__":
    my_list = SingleStatList()
    my_list.append(SingleStat("image1", 0.5, "needle", "IOU"))
    my_list.append(SingleStat("image2", -1.5, "needle", "IOU"))
    my_list.append(SingleStat("image3", 2.5, "needle", "IOU"))

    print(my_list)
    my_list.sort()
    print(my_list)
    my_list.sort(reverse=True)
    print(my_list)
