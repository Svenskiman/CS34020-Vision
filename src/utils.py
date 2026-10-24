"""Reusable utility functions"""

import cv2 as cv
import numpy as np
from pathlib import Path


def output_image_properties(image_path: str) -> None:
    """
    Print basic properties of an image.
    :param image_path: Path of the image
    :return: None
    """
    img = cv.imread(image_path)
    if img is None:
        print(f"Failed to load image at: {image_path}")
        return
    print(
        f"Properties of {image_path}:\n",
        f"Number of pixels: {img.size}\n",
        f"Datatype: {img.dtype}\n",
        f"Shape (rows, columns, channels) = {img.shape}"
    )


def output_image(
    image: np.ndarray,
    output_dir: str,
    name: str,
    display: bool = False
) -> None:
    """
    Save an image and optionally display it.
    :param image: The image to save
    :param output_dir: The directory to save the image
    :param name: Name of the image
    :param display: Display with cv.imshow if true
    :return: None
    """

    output_directory = Path(output_dir)
    # Make output dir if it doesn't exist
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{name}.png"

    if cv.imwrite(str(output_path), image):
        print(f"Image {name} saved at: {output_path}")
    else:
        print(f"Failed to write image at {output_path}")

    if display:
        cv.imshow(name, image)