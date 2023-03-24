from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from monai.networks.nets import FlexibleUNet

import torch


@dataclass
class AbstractInferencePipe(ABC):
    @abstractmethod
    def infer(self):
        pass

    @abstractmethod
    def upload_weights(self):
        pass


@dataclass
class FlexibleUnet1InferencePipe(AbstractInferencePipe):
    path_to_weights: Path
    device: str
    out_channels: int = 1

    def __post_init__(self):
        self.model = FlexibleUNet(
            in_channels=3,
            out_channels=self.out_channels,
            backbone="efficientnet-b0",
            pretrained=True,
            is_pad=False,
        ).to(self.device)

        self.upload_weights()

    def infer(self, im: torch.Tensor):
        im = im.to(self.device)
        # inferred = self.model(im[None]) > 0
        # im is a metatensor. im[None] is a tensor with a batch dimension of 1
        inferred = self.model(im[None].as_tensor())[0]
        inferred = inferred.argmax(dim=0, keepdim=True)
        inferred = inferred.detach().cpu()
        return inferred

    def upload_weights(self):
        self.model.load_state_dict(torch.load(self.path_to_weights))


if __name__ == "__main__":

    from monai.data import ThreadDataLoader, decollate_batch
    from surg_seg.Datasets.ImageDataset import ImageSegmentationDataset

    data_dir = Path("/home/juan1995/research_juan/accelnet_grant/data/rec03")
    ds = ImageSegmentationDataset(data_dir, "5colors")

    print(f'one-hot shape: {ds[100]["label"].shape}')
    dl = ThreadDataLoader(ds, batch_size=4, num_workers=0, shuffle=True)

    model = FlexibleUNet(
        in_channels=3,
        out_channels=5,
        backbone="efficientnet-b0",
        pretrained=True,
        is_pad=False,
    ).to("cpu")

    for batch_data in dl:
        inputs, labels = batch_data["image"], batch_data["label"]
        print(f"one-hot shape: {labels.shape}")

        inputs = inputs.to("cpu")
        model_out = model(inputs)
        print(f"model out shape: {model_out.shape}")

        break
