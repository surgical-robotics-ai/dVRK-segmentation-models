import numpy as np
import argparse
from pathlib import Path
import cv2
import click
import tqdm


@click.command()
@click.option("--infile", required=True, help="input file")
@click.option(
    "--outdir",
    default=None,
    help="Output directory. If not given will be saved in <infile>/<video-name>_images.",
)
@click.option("--sample", default=1, help="Sample every n frames.")
def images_from_video(infile, outdir, sample):
    """
    Convert video to images.
    """

    infile = Path(infile)

    if outdir is not None:
        outdir = Path(outdir)
    else:
        outdir = infile.parent
        outdir = outdir / (infile.with_suffix("").name + "_images")
    outdir.mkdir(exist_ok=True)

    if not infile.exists():
        print("input file not found")
        exit(0)

    cap = cv2.VideoCapture(str(infile))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error opening video file")
        exit()

    index = 0

    with tqdm.tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if index % sample == 0:
                    cv2.imwrite(str(outdir / f"image_{index:05d}.png"), frame)
            else:
                break

            index += 1
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()
    print("finish processing videos")


if __name__ == "__main__":
    images_from_video()
