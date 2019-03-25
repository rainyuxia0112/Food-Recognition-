import cv2
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
from __future__ import division
import os

from PIL import Image
def show(image):
    # Figure size in inches
    plt.figure(figsize=(15, 15))

    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')


def show_hsv(hsv):
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    show(rgb)

def show_mask(mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')


def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    show(img)

# read the image
os.mkdir('salad_final/')
for ele in os.listdir('salad/')[1:]:
    image = cv2.imread('salad/' +ele)
    img = Image.open('salad/' + ele)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images = []
    for i in [0, 1, 2]:
        colour = image.copy()
        if i != 0: colour[:, :, 0] = 0
        if i != 1: colour[:, :, 1] = 0
        if i != 2: colour[:, :, 2] = 0
        images.append(colour)

        # Convert from RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    images = []
    for i in [0, 1, 2]:
        colour = hsv.copy()
        if i != 0: colour[:, :, 0] = 0
        if i != 1: colour[:, :, 1] = 255
        if i != 2: colour[:, :, 2] = 255
        images.append(colour)

    hsv_stack = np.vstack(images)
    rgb_stack = cv2.cvtColor(hsv_stack, cv2.COLOR_HSV2RGB)

    # Blur image slightly
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)

    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))     # change it 绿色

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # Fill small gaps
    image_red_closed = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)

    # Remove specks
    image_red_closed_then_opened = cv2.morphologyEx(image_red_closed, cv2.MORPH_OPEN, kernel)

    big_contour, red_mask = find_biggest_contour(image_red_closed_then_opened)

    # Bounding ellipse
    image_with_ellipse = image.copy()
    x, y, w, h = cv2.boundingRect(big_contour)
    crpim = img.crop((0.9*x, 0.9*y, x + 1.1*w, y + 1.1*h)).resize((128, 128))
    crpim.save('salad_final/'+ ele)

