import cv2
from pathlib import Path
import natsort


def create_video_from_images(path_to_images: Path, fps: int = 4):
    """Create a video from a set of images."""

    if not path_to_images.exists():
        print(f"Path to images does not exist: {path_to_images}")

    output_name = path_to_images.parent.name + "_seg_" + path_to_images.name + ".avi"

    images_list = list(path_to_images.glob("*.png"))
    images_list = natsort.natsorted(images_list)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_out = cv2.VideoWriter(str(path_to_images / output_name), fourcc, fps, (640, 480))

    for image_path in images_list:
        print(image_path)
        img = cv2.imread(str(image_path))
        video_out.write(img)
    video_out.release()


def main():
    vid_root = Path(
        "/home/juan1995/research_juan/accelnet_grant/data/phantom2_data_processed/rec01"
    )

    create_video_from_images(vid_root / "raw")
    create_video_from_images(vid_root / "annotation2colors")
    create_video_from_images(vid_root / "annotation4colors")
    create_video_from_images(vid_root / "annotation5colors")


if __name__ == "__main__":
    main()
