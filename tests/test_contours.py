import cv2 as cv
import numpy as np
from pathlib import Path

from src.utils import output_image


def get_bounding_boxes(
    mask: np.ndarray,
    min_area: int = 10
) -> list[tuple[int, int, int, int]]:
    """
    Find the bounding boxes of the objects in a mask.
    :param mask: The binary mask to examine
    :param min_area: Smallest contour area to keep
    :return: List of (x, y, width, height) bounding boxes
    """

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        if cv.contourArea(contour) < min_area:
            continue
        boxes.append(cv.boundingRect(contour))

    return boxes


def test_contours(
    image_path: str,
    mask_path: str,
    min_area: int = 10,
    display: bool = False
) -> None:
    """
    Test contour detection by drawing bounding boxes on an image.
    :param image_path: The path of the base image to draw on
    :param mask_path: The path of the mask produced by an earlier stage
    :param min_area: Smallest contour area to keep
    :param display: Display with cv.imshow if true
    :return: None
    """

    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    image = cv.imread(image_path)
    if mask is None or image is None:
        print(f"Failed to load mask or image")
        return

    image_name = Path(image_path).stem # Return file name without extension
    output_dir = f"results/tests/contours/{image_name}"

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    kept = [
        contour for contour in contours
        if cv.contourArea(contour) >= min_area
    ]
    boxes = get_bounding_boxes(mask, min_area)
    print(f"Contours found: {len(contours)} (kept {len(boxes)} with area >= {min_area})")

    for i, (contour, (x, y, w, h)) in enumerate(zip(kept, boxes)):
        print(f"{i}: area = {cv.contourArea(contour)}, box = ({x}, {y}, {w}, {h})")
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    output_name = f"{Path(mask_path).stem.removesuffix('-Mask')}-Bounding-Boxes"
    output_image(image, output_dir, output_name, display)
