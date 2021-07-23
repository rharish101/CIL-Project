"""Turns the probaility output in a folder into 0/1."""
import glob

import cv2
from torchvision.io import read_image

THRESHOLD = 0.5

if __name__ == "__main__":
    for file_name in glob.glob("./*.png"):
        image = ((read_image(file_name).numpy() / 255) > THRESHOLD).astype(int)
        cv2.imwrite("../outputs" + file_name[1:], image[0] * 255)
