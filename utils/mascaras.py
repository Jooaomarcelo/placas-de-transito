import numpy as np
import cv2 as cv

def color_segmentation(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Vermelho
    lower_red1 = np.array([0, 130, 90])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 130, 90])
    upper_red2 = np.array([180, 255, 255])

    # Amarelo
    lower_yellow = np.array([22, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)

    mask_combined = cv.bitwise_or(mask_red1, mask_red2)
    mask_combined = cv.bitwise_or(mask_combined, mask_yellow)

    return mask_combined