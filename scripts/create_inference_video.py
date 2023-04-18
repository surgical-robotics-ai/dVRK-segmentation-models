from dataclasses import dataclass
from pathlib import Path
import cv2
import os
import numpy as np
import torch

from monai.data.video_dataset import VideoFileDataset
from monai.visualize.utils import blend_images
from tqdm import tqdm
from surg_seg.Datasets.VideoDatasets import CombinedVidDataset, VidDataset

# from surg_seg.Datasets.ImageDataset import ImageDataset

from surg_seg.Networks.Models import FlexibleUnet1InferencePipe, AbstractInferencePipe


@dataclass
class VideoCreator:
    fps: float

    def __post_init__(self):
        self.get_codec()
        self.fourcc = cv2.VideoWriter_fourcc(*self.codec)

    def create_video(
        self,
        model_pipe: FlexibleUnet1InferencePipe,
        output_file,
        ds: CombinedVidDataset,
        check_codec=True,
    ):
        if check_codec:
            self.check_codec

        print(f"{len(ds)} frames @ {self.fps} fps: {output_file}...")

        for idx in tqdm(range(len(ds))):
            img = ds[idx]["image"]
            inferred_single_ch = model_pipe.infer_from_monai_tensor(img)
            blended = blend_images(img, inferred_single_ch, cmap="viridis", alpha=0.8)

            if idx == 0:
                width_height = blended.shape[1:][::-1]
                video = cv2.VideoWriter(output_file, self.fourcc, self.fps, width_height)

            blended = (np.moveaxis(blended, 0, -1) * 254).astype(np.uint8)
            blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            video.write(blended)

        video.release()

        if not os.path.isfile(output_file):
            raise RuntimeError("video not created:", output_file)

        print("Success!")

    def get_codec(self):
        codecs = VideoFileDataset.get_available_codecs()
        self.codec, self.ext = next(iter(codecs.items()))
        print(self.codec, self.ext)

    def check_codec(self):
        codec_success = cv2.VideoWriter().open("test" + self.ext, self.fourcc, 1, (10, 10))
        if not codec_success:
            raise RuntimeError("failed to open video.")
        os.remove("test" + self.ext)


def config1():
    """Config when doing inference Annie's folder structure"""
    # path_to_weights = Path("./assets/weights/myweights_image_all_datasets/myweights.pt")
    path_to_weights = Path("assets/weights/myweights_3d_med_2_all_ds3/myweights.pt")

    ## Data loading
    rec_num = 1
    vid_root = Path(
        f"/home/juan1995/research_juan/accelnet_grant/data/phantom2_data_processed/rec{rec_num:02d}/"
    )
    vid_filepath = vid_root / f"raw/rec{rec_num:02d}_seg_raw.avi"

    output_path = vid_root / "inferred.mp4"

    return path_to_weights, vid_filepath, output_path


def config2():
    """Config with simple folder structure."""
    path_to_weights = Path("assets/weights/myweights_3d_med_2_all_ds3/myweights.pt")
    ## Data loading
    # vid_filepath = Path(
    #     "/home/juan1995/research_juan/accelnet_grant/data/dVRK_data_processed/rec03_right.avi"
    # )
    vid_filepath = Path(
        "/home/juan1995/research_juan/accelnet_grant/data/zed_camera_processed/rec04/2023-04-17_18-35-37_rightXX.avi"
    )
    output_path = vid_filepath.parent / (vid_filepath.with_suffix("").name + "_inferred.mp4")

    return path_to_weights, vid_filepath, output_path


def main():
    device = "cuda"
    # Choose which config to use config1() or config2()
    path_to_weights, vid_filepath, output_path = config2()

    model_pipe = FlexibleUnet1InferencePipe(path_to_weights, device, out_channels=5)
    ds = VidDataset(vid_filepath)

    # create video
    fps = ds.ds_img.get_fps()
    print(f"fps {fps}")
    video_creator = VideoCreator(fps)

    with torch.no_grad():
        video_creator.create_video(model_pipe, str(output_path), ds)


if __name__ == "__main__":
    main()
