import cv2 as cv
import numpy as np
from pathlib import Path

from src.utils import output_image
from tests.test_channels import get_channels


def get_mask(
    image: np.ndarray,
    threshold_method: str,
    threshold: int = 255,
    block_size: int = 101,
    c: int = -2,
) -> tuple[str, np.ndarray]:
    """
    Create an image mask with different threshold methods.
    :param image: The image to turn into mask
    :param threshold_method:
    :param threshold:
    :param block_size:
    :param c:
    :return: Tuple of name and the mask image
    """

    if threshold_method == "binary":
        _, mask = cv.threshold(
            image,
            threshold,
            255,
            cv.THRESH_BINARY
        )
        return f"Binary-{threshold}", mask

    if threshold_method == "otsu":
        # Threshold is set to 0 as Otsu picks its own
        otsu_threshold, mask = cv.threshold(
            image,
            0,
            255,
            cv.THRESH_BINARY + cv.THRESH_OTSU
        )
        return f"Otsu-{int(otsu_threshold)}", mask

    if threshold_method == "adaptive":
        mask = cv.adaptiveThreshold(
            image,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            blockSize=block_size,
            C=c,
        )
        return f"Adaptive-{block_size}-({c})", mask

    raise ValueError(f"Unknown threshold method: {threshold_method}")


def test_thresholding(
    image_path: str,
    colour_space: str,
    channel: str,
    threshold_methods: list[str],
    threshold: int = 255,
    block_size: int = 101,
    c: int = -2,
    display: bool = False,
) -> None:
    """
    Test different thresholding methods for an image channel.
    :param image_path: Path of the base image
    :param colour_space: Colour space to use
    :param channel: Channel to use
    :param threshold_methods: Threshold methods to use
    :param threshold:
    :param block_size:
    :param c:
    :param display:
    :return: None
    """

    image = cv.imread(image_path)
    if image is None:
        print(f"Failed to load image at: {image_path}")
        return

    image_name = Path(image_path).stem
    output_dir = f"results/tests/thresholding/{image_name}"

    channel_image = get_channels(image, colour_space)[channel]

    for threshold_method in threshold_methods:
        mask_name, mask = get_mask(channel_image, threshold_method, threshold, block_size, c)
        result = cv.bitwise_and(image, image, mask=mask)

        output_image(mask, output_dir, f"{channel}-{mask_name}-Mask", display)
        output_image(result, output_dir, f"{channel}-{mask_name}-Result", display)