from typing import Dict, List
import torch
import pytest
from pathlib import Path
from PIL import Image
from natsort import natsorted
import numpy as np
from monai.metrics.meaniou import compute_iou, MeanIoU
from surg_seg.Datasets.SegmentationLabelParser import YamlSegMapReader, SegmentationLabelParser

data_path = Path(__file__).parent / "./Data/"


def img2tensor(paths: List[Path]):
    list_of_tensors = []
    for path in paths:
        img = np.array(Image.open(path))
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        list_of_tensors.append(torch.tensor(img))

    list_of_tensors = np.concatenate(list_of_tensors, axis=0)
    return torch.tensor(list_of_tensors)


def label2tensor(paths: List[Path], label_parser: SegmentationLabelParser):
    list_of_tensors = []
    for path in paths:
        label = np.array(Image.open(path))
        label = label_parser.convert_rgb_to_onehot(label)
        label = np.expand_dims(label, axis=0)
        list_of_tensors.append(torch.tensor(label))

    list_of_tensors = np.concatenate(list_of_tensors, axis=0)
    return torch.tensor(list_of_tensors)


@pytest.fixture
def label_parser() -> SegmentationLabelParser:
    seg_map_reader = YamlSegMapReader(data_path / "dataset_config.yaml")
    label_parser = SegmentationLabelParser(seg_map_reader)

    return label_parser


@pytest.fixture
def sample_data(label_parser: SegmentationLabelParser):

    raw_img = data_path / "raw"
    raw_img = natsorted(list(raw_img.glob("*.png")))
    raw_img = img2tensor(raw_img)

    label = data_path / "label"
    label = natsorted(list(label.glob("*.png")))
    label = label2tensor(label, label_parser)

    return {"raw": raw_img, "label": label}


def test_sample_data_path():
    assert data_path.exists(), f"{data_path} does not exist"


def test_raw_img_dimensions(sample_data):
    raw_img = sample_data["raw"]
    assert raw_img.shape == (3, 3, 1024, 1280), "Raw image shape is {}".format(raw_img.shape)


def test_label_dimensions(sample_data):
    label = sample_data["label"]
    assert label.shape == (3, 3, 1024, 1280), "Label shape is {}".format(label.shape)


def test_label_parser(sample_data, label_parser: SegmentationLabelParser):
    assert label_parser.mask_num == 3, "Mask num is {}".format(label_parser.mask_num)


def test_single_label_iou_dimensions(sample_data):
    label = sample_data["label"]

    one_hotlabel = torch.unsqueeze(label[0], 0)
    iou_tensor = compute_iou(one_hotlabel, one_hotlabel, include_background=False)
    assert iou_tensor.shape == (1, 2), "IOU tensor shape is {}".format(iou_tensor.shape)


def test_single_label_iou_value(sample_data: Dict[str, torch.Tensor]):
    label = sample_data["label"]
    one_hotlabel = torch.unsqueeze(label[0], 0)
    iou_tensor = compute_iou(one_hotlabel, one_hotlabel, include_background=False)

    answer = torch.tensor([[1.0, 1.0]])
    assert torch.allclose(iou_tensor, answer), "IOU tensor value is {}".format(iou_tensor)


def test_batch_label_iou_value(sample_data: Dict[str, torch.Tensor]):
    label = sample_data["label"]
    iou_tensor = compute_iou(label, label, include_background=False)

    answer = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    assert torch.allclose(iou_tensor, answer), "IOU tensor value is {}".format(iou_tensor)


def test_MeanIOU_class(sample_data: Dict[str, torch.Tensor]):

    batch_label = sample_data["label"]

    iou_metric = MeanIoU(include_background=False, reduction="mean")

    batch_iou = iou_metric(y_pred=batch_label, y=batch_label)
    epoch_iou = iou_metric.aggregate().item()
    assert np.isclose(epoch_iou, 1.0), "IOU metric value is {}".format(epoch_iou)
    iou_metric.reset()


def test_MeanIOU_get_buffer_method_with_single_batch(sample_data: Dict[str, torch.Tensor]):

    batch_label = sample_data["label"]
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    batch_iou = iou_metric(y_pred=batch_label, y=batch_label)

    epoch_iou = iou_metric.get_buffer()

    assert epoch_iou.shape == (3, 2), f"IOU metric buffer shape is {epoch_iou.shape}"

    iou_metric.reset()


def test_MeanIOU_get_buffer_method_with_mult_batch(sample_data: Dict[str, torch.Tensor]):

    batch_label = sample_data["label"]

    batch1 = torch.unsqueeze(batch_label[0], 0)
    batch2 = batch_label[1:]

    iou_metric = MeanIoU(include_background=False, reduction="mean")

    for b in [batch1, batch2]:
        batch_iou = iou_metric(y_pred=b, y=b)

    epoch_iou = iou_metric.get_buffer()
    assert epoch_iou.shape == (3, 2), f"IOU metric buffer shape is {epoch_iou.shape}"

    iou_metric.reset()


def test_MeanIOU_get_buffer_method_with_background(sample_data: Dict[str, torch.Tensor]):

    batch_label = sample_data["label"]
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    batch_iou = iou_metric(y_pred=batch_label, y=batch_label)

    assert batch_iou.shape == (3, 3), f"IOU metric buffer shape is {batch_iou.shape}"
    iou_metric.reset()
