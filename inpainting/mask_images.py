"""Cropping and masking images

This script creates cropped and masked images that can be used for generating new training data using RePaint.
Images are cropped to 256x256 as required by RePaint. Cropped images and masks are saved since they are needed as
ground truth and mask, respectively, for RePaint. The masked images are not needed for RePaint, but saved in case they
are needed for other purposes.
"""

import os
import re
import pickle
from glob import glob
import numpy as np
from PIL import Image
from random import randrange

PATH = "images"
CROPPED_PATH = "cropped_images"
MASK_PATH = "masks"
MASKED_PATH = "cropped_and_masked_images"

NUM_CROPS = 4
CROP_SIZE = 256

MASK_BOX_HEIGHT = 50
MASK_BOX_WIDTH = 70
NUM_MASK_BOXES_PER_IMAGE = 3


def mask_images():
    crop_positions = np.zeros((len(glob(PATH + '/*.png')), NUM_CROPS, 2), dtype=int)
    for f in glob(PATH + '/*.png'):
        img = Image.open(f)
        img_idx = int(re.search(r'\d+', f).group()) # first number in string
        x, y = img.size

        for crop_idx in range(NUM_CROPS):
            x1 = randrange(0, x - CROP_SIZE)
            y1 = randrange(0, y - CROP_SIZE)
            cropped_img = img.crop((x1, y1, x1 + CROP_SIZE, y1 + CROP_SIZE))
            crop_positions[img_idx][crop_idx][0] = x1
            crop_positions[img_idx][crop_idx][1] = y1

            path = os.path.join(CROPPED_PATH, f"satimage_{img_idx}_crop_{crop_idx}.png")
            cropped_img.save(path)

            cropped_img_arr = np.asarray(cropped_img).copy()
            mask = 255 * np.ones_like(cropped_img_arr[:,:,:3])
            for i in range(NUM_MASK_BOXES_PER_IMAGE):
                x1 = randrange(0, CROP_SIZE - MASK_BOX_WIDTH)
                y1 = randrange(0, CROP_SIZE - MASK_BOX_HEIGHT)
                mask[y1:(y1 + MASK_BOX_HEIGHT), x1:(x1 + MASK_BOX_WIDTH), :] = 0
                cropped_img_arr[y1:(y1 + MASK_BOX_HEIGHT), x1:(x1 + MASK_BOX_WIDTH), :] = 0
            mask = Image.fromarray(mask)

            path = os.path.join(MASK_PATH, f"satimage_{img_idx}_mask_{crop_idx}.png")
            mask.save(path)

            masked_img = Image.fromarray(cropped_img_arr)

            path = os.path.join(MASKED_PATH, f"satimage_{img_idx}_cropped_and_masked_{crop_idx}.png")
            masked_img.save(path)

    file = open("crop_positions.pkl", "wb")
    pickle.dump(crop_positions, file)
    file.close()


if __name__ == "__main__" :
    mask_images()