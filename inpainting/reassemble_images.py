"""Combining crops with the full image

This script combines (edited) crops of an image with the full image.
"""

import os
import re
import pickle
from glob import glob
from PIL import Image


FULL_IMAGES_PATH = "images"
CROPS_MASKED_PATH = "cropped_and_masked_images"
CROPS_INPAINTED_PATH = "cropped_and_inpainted_images"
MASKED_PATH = "masked_images"
INPAINTED_PATH = "inpainted_images"


def reassemble_image(filename, crop_positions, save_path, title):
    crop = Image.open(filename)

    indices = re.findall(r'\d+', filename)
    img_idx = int(indices[0])
    crop_idx = int(indices[1])

    path = os.path.join(FULL_IMAGES_PATH, f"satimage_{img_idx}.png")
    full_img = Image.open(path)

    full_img.paste(crop, tuple(crop_positions[img_idx][crop_idx]))

    path = os.path.join(save_path, f"satimage_{title}_{img_idx}_{crop_idx}.png")
    full_img.save(path)


def main():
    file = open("crop_positions.pkl", "rb")
    crop_positions = pickle.load(file)
    file.close()

    for f in glob(CROPS_MASKED_PATH + '/*.png'):
        reassemble_image(f, crop_positions, MASKED_PATH, "masked")

    for f in glob(CROPS_INPAINTED_PATH + '/*.png'):
        reassemble_image(f, crop_positions, INPAINTED_PATH, "inpainted")

if __name__ == "__main__" :
    main()