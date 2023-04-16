import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import monai
from monai.visualize.utils import blend_images
from monai.data import ThreadDataLoader, decollate_batch
from surg_seg.Datasets.ImageDataset import ImageSegmentationDataset
from surg_seg.Datasets.VideoDatasets import CombinedVidDataset
from surg_seg.Networks.Models import FlexibleUnet1InferencePipe


def main():
    # load data
    root = Path("/home/juan1995/research_juan/accelnet_grant/data/phantom2_data_processed/")
    train_dirs = [root / "rec01", root / "rec03", root / "rec05"]
    val_dirs = [root / "rec02", root / "rec04"]
    ds = ImageSegmentationDataset(val_dirs, "5colors")

    # load model
    device = "cuda"
    path_to_weights = Path("./assets/weights/myweights_3d_med_2_all_ds3/myweights.pt")
    model_pipe = FlexibleUnet1InferencePipe(path_to_weights, device, out_channels=5)

    print(f"Number of frames in dataset: {len(ds)}")
    nrow, ncol = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3), facecolor="white")
    nexamples = nrow
    frames = sorted(np.random.choice(len(ds), size=nexamples, replace=False))
    for idx, frame in enumerate(frames):
        ax = axes[idx, :]
        _ds = ds[frame]
        inference = model_pipe.infer_from_transformed_tensor(_ds["image"].to(model_pipe.device))
        inference = inference.detach().cpu()

        # Inference
        img, lbl = _ds["image"], _ds["label"]
        blended = blend_images(img, inference, cmap="viridis", alpha=0.8).numpy()
        blended = (np.transpose(blended, (1, 2, 0)) * 254).astype(np.uint8)
        ax[0].imshow(blended)
        ax[0].set_title(f"inference: {frame}")
        ax[0].axis("off")

        # Ground truth
        lbl = torch.argmax(lbl, axis=0)
        lbl = torch.unsqueeze(lbl, 0)
        blended = blend_images(img, lbl, cmap="viridis", alpha=0.8).numpy()
        blended = (np.transpose(blended, (1, 2, 0)) * 254).astype(np.uint8)
        ax[1].imshow(blended)
        ax[1].set_title(f"Ground truth: {frame}")
        ax[1].axis("off")

    fig.set_tight_layout(True)

    plt.show()


if __name__ == "__main__":
    main()
