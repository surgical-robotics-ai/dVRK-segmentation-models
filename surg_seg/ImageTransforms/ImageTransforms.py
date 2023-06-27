from __future__ import annotations
import random
import torch
import torchvision.transforms as T
import monai.transforms as mt
from torchvision import transforms
from torchvision.transforms import functional as TF


class ImageTransforms:

    img_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet normalize
        ]
    )

    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    inv_transforms = T.Compose(
        [
            T.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )

    def geometric_transforms(image: torch.Tensor, mask: torch.Tensor):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(480, 640))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask

    predictions_transforms = mt.Compose(
        [
            mt.Activations(sigmoid=True),
            mt.AsDiscrete(threshold=0.5),
        ]
    )
