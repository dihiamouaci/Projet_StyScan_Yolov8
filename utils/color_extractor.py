import cv2
import numpy as np

COLOR_RANGES = {
    "Black": ([0, 0, 0], [180, 255, 50]),
    "White": ([0, 0, 200], [180, 40, 255]),
    "Red": ([0, 100, 50], [10, 255, 255]),
    "Blue": ([90, 50, 50], [130, 255, 255]),
    "Green": ([35, 50, 50], [85, 255, 255]),
    "Yellow": ([20, 100, 100], [35, 255, 255]),
}


def extract_color(image):
    if image is None or image.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    detected_color = "Unknown"
    max_pixels = 0

    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        pixels = cv2.countNonZero(mask)

        if pixels > max_pixels:
            max_pixels = pixels
            detected_color = color_name

    return detected_color
