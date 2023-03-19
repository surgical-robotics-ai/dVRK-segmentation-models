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


class CombinedVidDataset(Dataset):
    def __init__(self, vid_file, seg_file):
        self.ds_img = VideoFileDataset(str(vid_file), Transforms.vid_transforms)
        self.ds_lbl = VideoFileDataset(str(seg_file), Transforms.seg_transforms)

    def __len__(self):
        return len(self.ds_img)

    def __getitem__(self, idx):
        return {"image": self.ds_img[idx], "label": self.ds_lbl[idx]}
