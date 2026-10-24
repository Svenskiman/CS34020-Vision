import cv2 as cv
import numpy as np
from pathlib import Path

from src.utils import output_image


def get_channels(
    image: np.ndarray,
    colour_space: str
) -> dict[str, np.ndarray]:
    """
    Convert the provided image into a new colour space
    :param image: The image to convert
    :param colour_space: The colour space to use
    :return: Dict of images for the colour space and its channels
    """

    if colour_space == "RGB":
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        red, green, blue = cv.split(rgb_image)
        return {
            "RGB": rgb_image,
            "RGB-Red": red,
            "RGB-Green": green,
            "RGB-Blue": blue,
        }

    if colour_space == "HSV":
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        hue, saturation, value = cv.split(hsv_image)
        return {
            "HSV": hsv_image,
            "HSV-Hue": hue,
            "HSV-Saturation": saturation,
            "HSV-Value": value,
        }

    if colour_space == "LAB":
        lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        brightness, red_green, blue_yellow = cv.split(lab_image)
        return {
            "LAB": lab_image,
            "LAB-Brightness (L)": brightness,
            "LAB-Red-Green (A)": red_green,
            "LAB-Blue-Yellow (B)": blue_yellow,
        }

    raise ValueError(f"Unknown colour space: {colour_space}")


def test_channels(
    image_path: str,
    colour_spaces: list[str],
    display: bool = False
) -> None:
    """
    Test different colour spaces and channel results on
    an image.
    :param image_path: The path of the image to test
    :param colour_spaces: List of colour spaces to convert to
    :param display: Display with cv.imshow if true
    :return: None
    """

    image = cv.imread(image_path)
    if image is None:
        print(f"Failed to load image at: {image_path}")
        return

    image_name = Path(image_path).stem
    output_dir = f"results/tests/channels/{image_name}"

    for colour_space in colour_spaces:
        channels = get_channels(image, colour_space)
        for channel_name in channels:
            output_image(channels[channel_name], output_dir, channel_name, display)