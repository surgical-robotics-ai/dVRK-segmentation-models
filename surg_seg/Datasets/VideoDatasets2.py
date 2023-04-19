from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from torch.utils.data import Dataset, IterableDataset
from monai.visualize.utils import blend_images
from monai.utils.enums import ColorOrder
from monai.utils.module import optional_import
from tqdm import tqdm
import monai.transforms as mt

class VideoDataset:
    # import inside class to avoid webcam blinking on ``import monai``.

    def __init__(
        self,
        video_source: str,
        transform: Callable,
        max_num_frames: int | None = None,
        color_order: str = ColorOrder.RGB,
        multiprocessing: bool = False,
        channel_dim: int = 0,
    ) -> None:

        if color_order not in ColorOrder:
            raise NotImplementedError

        self.color_order = color_order
        self.channel_dim = channel_dim
        self.video_source = video_source
        self.multiprocessing = multiprocessing
        if not multiprocessing:
            self.cap = self.open_video(video_source)
        self.transform = transform
        self.max_num_frames = max_num_frames

    @staticmethod
    def open_video(video_source: str):
        """
        Use OpenCV to open a video source from either file or capture device.
        Args:
            video_source: filename or index referring to capture device.
        Raises:
            RuntimeError: Source is a file but file not found.
            RuntimeError: Failed to open source.
        """
        if isinstance(video_source, str) and not os.path.isfile(video_source):
            raise RuntimeError("Video file does not exist: " + video_source)
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_source}")
        return cap

    def _get_cap(self):
        """Return the cap. If multiprocessing, create a new one. Else return the one from construction time."""
        return self.open_video(self.video_source) if self.multiprocessing else self.cap

    def get_fps(self) -> int:
        """Get the FPS of the capture device."""
        return self._get_cap().get(cv2.CAP_PROP_FPS)  # type: ignore

    def get_frame(self):
        """Get next frame. For a file, this will be the next frame, whereas for a camera
        source, it will be the next available frame."""
        ret, frame = self._get_cap().read()
        if not ret:
            raise RuntimeError("Failed to read frame.")
        # Switch color order if desired
        if self.color_order == ColorOrder.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # move channel dim
        frame = np.moveaxis(frame, -1, self.channel_dim)
        return self.transform(frame) if self.transform is not None else frame


class VideoFileDataset(Dataset, VideoDataset):
    """
    Video dataset from file.
    This class requires that OpenCV be installed.
    """

    def __init__(self, *args, **kwargs) -> None:
        VideoDataset.__init__(self, *args, **kwargs)
        num_frames = self.get_num_frames()
        self.max_num_frames = num_frames

    @staticmethod
    def get_available_codecs() -> dict[str, str]:
        """Try different codecs, see which are available.
        Returns a dictionary with of available codecs with codecs as keys and file extensions as values."""
        all_codecs = {"mp4v": ".mp4", "X264": ".avi", "H264": ".mp4", "MP42": ".mp4", "MJPG": ".mjpeg", "DIVX": ".avi"}
        codecs = {}
        writer = cv2.VideoWriter()
        with tempfile.TemporaryDirectory() as tmp_dir:
            for codec, ext in all_codecs.items():
                fname = os.path.join(tmp_dir, f"test{ext}")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                noviderr = writer.open(fname, fourcc, 1, (10, 10))
                if noviderr:
                    codecs[codec] = ext
        writer.release()
        return codecs

    def get_num_frames(self) -> int:
        """
        Return the number of frames in a video file.
        Raises:
            RuntimeError: no frames found.
        """
        num_frames = int(self._get_cap().get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames == 0:
            raise RuntimeError("0 frames found")
        return num_frames

    def __len__(self):
        return self.max_num_frames

    def __getitem__(self, index: int):
        """
        Fetch single data item from index.
        """
        if self.max_num_frames is not None and index >= self.max_num_frames:
            raise IndexError
        self._get_cap().set(cv2.CAP_PROP_POS_FRAMES, index)
        return self.get_frame()

class Transforms:
    vid_transforms = mt.Compose(
        [
            mt.DivisiblePad(32),
            mt.ScaleIntensity(),
            mt.CastToType(torch.float32),
        ]
    )
    seg_transforms = mt.Compose([vid_transforms, mt.Lambda(lambda x: x[:1])]) 
class VidDataset(Dataset):
    def __init__(self, vid_file):
        self.ds_img = VideoFileDataset(str(vid_file), Transforms.vid_transforms)

    def __len__(self):
        return len(self.ds_img)

    def __getitem__(self, idx):
        return {"image": self.ds_img[idx]}

class CombinedVidDataset(Dataset):
    def __init__(self, vid_file, seg_file):
        self.ds_img = VideoFileDataset(str(vid_file), Transforms.vid_transforms)
        self.ds_lbl = VideoFileDataset(str(seg_file), Transforms.seg_transforms)

        self.label_channels = 1

    def __len__(self):
        return len(self.ds_img)

    def __getitem__(self, idx):
        return {"image": self.ds_img[idx], "label": self.ds_lbl[idx]}
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

