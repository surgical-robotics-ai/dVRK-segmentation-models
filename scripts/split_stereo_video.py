import numpy as np
import argparse
from pathlib import Path
import cv2
import click
import tqdm


@click.command()
@click.option("--infile", required=True, help="input file")
@click.option(
    "--outfile",
    default=None,
    help="Output file. If not given will be saved in the same directory as the input file.",
)
def split_stereo_video(infile, outfile):
    """
    Script to split stereo video into left/right videos.
    """

    infile = Path(infile)

    if outfile is not None:
        outfile = Path(outfile)
    else:
        outfile = infile.parent
    outfile.mkdir(exist_ok=True)

    if not infile.exists():
        print("input file not found")
        exit(0)

    cap = cv2.VideoCapture(str(infile))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error opening video file")
        exit()

    enc = cv2.VideoWriter_fourcc(*"XVID")
    vid_format = ".avi"
    left_out_path = outfile / (infile.with_suffix("").name + "_left" + vid_format)
    right_out_path = outfile / (infile.with_suffix("").name + "_right" + vid_format)
    print(f"writing to {left_out_path}")

    left_out = None
    right_out = None
    index = 0

    with tqdm.tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if index == 0:
                frame_width = frame.shape[1] // 2
                frame_height = frame.shape[0]
                left_out = cv2.VideoWriter(str(left_out_path), enc, 25, (frame_width, frame_height))
                right_out = cv2.VideoWriter(
                    str(right_out_path), enc, 25, (frame_width, frame_height)
                )
            if ret:
                left = frame[:, :frame_width, :]
                right = frame[:, frame_width:, :]
                left_out.write(left)
                right_out.write(right)
            else:
                break
            index += 1
            pbar.update(1)

    cap.release()
    left_out.release()
    right_out.release()
    cv2.destroyAllWindows()

    print("finish processing videos")


if __name__ == "__main__":
    split_stereo_video()
