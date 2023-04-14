import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import monai
from monai.visualize.utils import blend_images
from monai.data import ThreadDataLoader, decollate_batch
from surg_seg.Datasets.VideoDatasets import CombinedVidDataset


# vid_root = Path("./assets/data/endo_vid/")
# vid_filepath = vid_root /"endo_vid.mp4"
# seg_filepath = vid_root /"endo_seg.mp4"

# vid_root = Path("./assets/data/ambf_vid/")
# vid_filepath = vid_root / "endo_vid.avi"
# seg_filepath = vid_root / "endo_seg.avi"

vid_root = Path("/home/juan1995/research_juan/accelnet_grant/data/phantom2_data_processed/rec01/")
vid_filepath = vid_root / "raw/rec01_seg_raw.avi"
seg_filepath = vid_root / "annotation4colors/rec01_seg_annotation4colors.avi"


def main():

    ds = CombinedVidDataset(vid_filepath, seg_filepath)
    dl = ThreadDataLoader(ds, batch_size=4, num_workers=0, shuffle=True)

    print(f"Number of frames in vid: {len(ds)}")
    nrow, ncol = 2, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3), facecolor="white")
    nexamples = nrow * ncol
    frames = sorted(np.random.choice(len(ds), size=nexamples, replace=False))
    for frame, ax in zip(frames, axes.flatten()):
        _ds = ds[frame]
        img, lbl = _ds["image"], _ds["label"]
        blended = blend_images(img, lbl, cmap="viridis", alpha=0.2)
        blended = np.moveaxis(blended, 0, -1)  # RGB to end
        ax.imshow(blended)
        ax.set_title(f"Frame: {frame}")
        ax.axis("off")
    fig.set_tight_layout(True)

    plt.show()


if __name__ == "__main__":
    main()
