import cv2 as cv
import numpy as np
import sys
import os

class MaskMapping:
    def __init__(self):
        """Match each BGR color in the original annotation to its mask counterpart.
        CHOICES:
        54: PSM dark grey
            - 125:  arm joint
            - 89:   gripper light grey
            - 87:   needle
            - 0:    thread
            - 104:  background
        """

        self.color_scheme = [54, 125, 89, 87, 2]

        self.mask = [
            {54: 225, 125: 225, 89: 225, 87: 225, 2: 225},
            {54: 225, 125: 225, 89: 225, 87: [0, 0, 250], 2: [0, 250, 0]},
            {54: 225, 125: 225, 89: [250, 0, 0], 87: [0, 0, 250], 2: [0, 250, 0]}
        ]


class ImageCustom:

    def __init__(self, img_name):
        self._image = cv.imread(img_name)

        if self._image is None:
            sys.exit('Error: can not load images')

        self._name = img_name
        self._row, self._col = self._image.shape[:2]
        self._annotation = self.get_ambfAnnotation()

        self.seg_str = self.__seg_name(img_name)

        self.img_map = MaskMapping()

    def get_width(self):
        return int(self._col / 2)

    def get_height(self):
        return self._row

    def __str__(self):
        """
        :return: image file name
        """
        return self._name

    def get_image(self):
        return self._image

    def __seg_name(self, _name):
        """
        name in format 2023-xx-xx_xx-xx-xx_0000xxx.png
        :param _name: image name; class variable
        :return: date, time, number
        """
        file_name = self._name
        num = file_name.strip('.png')[-7:]
        return num

    def get_rawImage(self):
        return self._image[:, : self.get_width(), :]

    def get_ambfAnnotation(self):
        return self._image[:, self.get_width():, :]

    def __get_annotation_pixel(self, x, y):
        """
        identify the instrument, thread, and needle according to BGR color for a single pixel
        :param x: the row of the pixel
        :param y: the col of the pixel
        :return: the BGR value of the pixel or -1 if it is part of the background
        """
        B, G, R = self._annotation[x, y, :]

        for color in self.img_map.color_scheme:
            if B == color and G == color and R == color:
                return color
        return -1

    def get_annotation(self, choice):
        """
        #1: black-white (PSM arms/grippers, needle, thread = white [0,1,2,3,4]), the rest are black
        #2: black-3 colors (PSM arms/grippers = white[0,1,2], needle = blue[3], thread = green[4]), the rest are black
        #3: black-4 colors (PSM arms = white[0,1], PSM grippers = purple[2], needle = blue[3], thread = green[4]), the rest are black
        :param choice: 1, 2, 3
        :return: New frame-->cv::Mat
        """
        if choice > 2:
            sys.exit('Error: Invalid annotation type')

        w = self.get_width()
        h = self.get_height()

        new_frame = np.zeros((h, w, 3), dtype="uint8")

        for i in range(0, h):
            for j in range(0, w):
                ans = self.__get_annotation_pixel(i, j)
                if ans == -1:
                    new_frame[i, j] = [0, 0, 0]
                else:
                    mask = self.img_map.mask[choice][ans]
                    if not isinstance(mask, list):
                        mask = [mask, mask, mask]
                    new_frame[i, j] = mask
        return new_frame


if __name__ == "__main__":

    img_name= os.path.join(os.getcwd().strip('annotation_reformat'), 'data/rec01/2023-02-16_13-03-37_0000000.png')
    img = ImageCustom(img_name)
    raw_img = img.get_rawImage()
    ambf_img = img.get_ambfAnnotation()
    annotation_0 = img.get_annotation(0)
    annotation_1 = img.get_annotation(1)
    annotation_2 = img.get_annotation(2)
    cv.imwrite('ambf_raw.png', raw_img)
    cv.imwrite('ambf.png', ambf_img)
    cv.imwrite('1.png', annotation_0)
    cv.imwrite('2.png', annotation_1)
    cv.imwrite('3.png', annotation_2)
