# import relevant classes
import sys
import os
import cv2 as cv
import glob
import argparse
from surg_seg.Utils.LabelGenerator import LabelGenerator  # customized image class

# ---------------------------------------------
# parse arguments
# ---------------------------------------------
argv = sys.argv
parser = argparse.ArgumentParser(
    description="find the location and the correct label of the to-be-processed dataset"
)

parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    required=True,
    help="path to the folder that contains the image folder, i.e. ~/data/rec01",
)
parser.add_argument(
    "-o",
    "--out_dir",
    type=str,
    required=False,
    default=None,
    help="output directory, default is current working directory",
)
args = parser.parse_args()

# ---------------------------------------------
# output path setup
# ---------------------------------------------
if args.out_dir is not None:
    folder_name = os.path.join(args.out_dir, "Output")
else:
    folder_name = os.path.join(os.getcwd(), "Output")
print("Output folder: ", folder_name)
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

sub_folder_name = os.path.join(os.getcwd(), os.path.join("Output", args.input_dir[-5:]))
if not os.path.exists(sub_folder_name):
    os.mkdir(sub_folder_name)

category = ["annotation2colors", "annotation4colors", "annotation5colors", "raw", "annotationAMBF"]
category_folder_name = list()
for catg in category:
    category_folder_name.append(
        os.path.join(os.getcwd(), os.path.join("Output", os.path.join(args.input_dir[-5:], catg)))
    )
    if not os.path.exists(category_folder_name[-1]):
        os.mkdir(category_folder_name[-1])

# ---------------------------------------------
# output file setup
# ---------------------------------------------
width, height = 1280, 480
frameSize = (int(width / 2), height)
filenames = glob.glob(
    os.path.join(os.getcwd().strip("annotation_reformat"), os.path.join(args.input_dir, "*.png"))
)
filenames.sort()
fourcc = cv.VideoWriter_fourcc(*"DIVX")
print("-- Initiation complete, start annotating...")


# ---------------------------------------------
# start processing raw images
# ---------------------------------------------
for x in range(5):
    vid = cv.VideoWriter(
        os.path.join(category_folder_name[x], f"{args.input_dir[-5:]}_seg_{category[x]}.avi"),
        fourcc,
        2,
        frameSize,
    )

    counter = 0
    total = len(filenames) / 10 + 1

    for img in filenames:
        cur_img = LabelGenerator(img)
        un_select = int(cur_img.seg_str) % 10 != 0
        if un_select:
            continue

        if x == 3:
            new_img = cur_img.get_rawImage()
        elif x == 4:
            new_img = cur_img.get_ambfAnnotation()
        else:
            new_img = cur_img.get_annotation(x)

        output_name = str("%06d" % int(int(cur_img.seg_str) / 10))
        cv.imwrite(os.path.join(category_folder_name[x], "img_{}.png".format(output_name)), new_img)
        vid.write(new_img)

        counter += 1
        sys.stdout.write(
            "\r-- Annotation %d / 5 | Progress %02.1f%%"
            % (x + 1, float(counter) / float(total) * 100.0)
        )
        sys.stdout.flush()

    vid.release()

print("\n Annotation formatting complete!")
