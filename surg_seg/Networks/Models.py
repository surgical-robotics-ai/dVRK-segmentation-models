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

    def __post_init__(self):
        self.model = FlexibleUNet(
            in_channels=3,
            out_channels=1,
            backbone="efficientnet-b0",
            pretrained=True,
            is_pad=False,
        ).to(self.device)

        self.upload_weights()

    def infer(self, im: torch.Tensor):
        im = im.to(self.device)
        inferred = self.model(im[None]) > 0
        inferred = inferred[0].detach().cpu()
        return inferred

    def upload_weights(self):
        self.model.load_state_dict(torch.load(self.path_to_weights))
