from pathlib import Path
from torch.utils.data import Dataset
import torch
from monai.data.video_dataset import VideoFileDataset
import monai.transforms as mt


class Transforms:
    vid_transforms = mt.Compose(
        [
            mt.DivisiblePad(32),
            mt.ScaleIntensity(),
            mt.CastToType(torch.float32),
        ]
    )
    seg_transforms = mt.Compose([vid_transforms, mt.Lambda(lambda x: x[:1])])  # rgb -> 1 chan


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


if __name__ == "__main__":

    vid_root = Path("/home/juan1995/research_juan/accelnet_grant/data/rec03/")

    vid_filepath = vid_root / "raw/rec03_seg_raw.avi"
    seg_filepath = vid_root / "annotation2colors/rec03_seg_annotation2colors.avi"

    ds = CombinedVidDataset(vid_filepath, seg_filepath)

    print(f"{ds[0]['image'].shape}")
    print(f"image dtype: {type(ds[0]['image'])}")
    print(f"image max values: {ds[0]['image'].max()}")
