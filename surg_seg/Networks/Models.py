from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from monai.networks.nets import FlexibleUNet
from monai.data.meta_tensor import MetaTensor
import torch
import numpy as np

from surg_seg.Datasets.ImageDataset import ImageTransforms


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

    def infer_from_monai_tensor(self, im: MetaTensor):
        """
        Infer from a monai tensor. This is the default output of all monai datasets.
        """
        im = im.to(self.device)
        # inferred = self.model(im[None]) > 0
        # im is a metatensor. im[None] is a tensor with a batch dimension of 1
        inferred = self.model(im[None].as_tensor())[0]
        inferred = inferred.argmax(dim=0, keepdim=True)
        inferred = inferred.detach().cpu()
        return inferred

    def infer_from_transformed_tensor(self, input_tensor: torch.Tensor):
        input_tensor = torch.unsqueeze(input_tensor, 0)  # Add batch dimension. 4D input_tensor
        inferred = self.model(input_tensor)
        inferred = inferred[0]  # go back to 3D tensor
        inferred_single_ch = inferred.argmax(dim=0, keepdim=True)  # Get a single channel image

        return inferred_single_ch

    def infer(self, im: np.ndarray):
        """
        Infer from RGB and unormalize numpy array. If loading data with opencv,
        make sure to convert to RGB.
        """
        input_tensor = ImageTransforms.img_transforms(im).to(self.device)
        input_tensor = torch.unsqueeze(input_tensor, 0)  # Add batch dimension. 4D input_tensor
        inferred = self.model(input_tensor)
        inferred = inferred[0]  # go back to 3D tensor
        inferred_single_ch = inferred.argmax(dim=0, keepdim=True)  # Get a single channel image

        return input_tensor, inferred_single_ch

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
