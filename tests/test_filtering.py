import cv2 as cv
import numpy as np
from pathlib import Path

from src.utils import output_image
from tests.test_channels import get_channels
from tests.test_thresholding import get_mask


def get_blur(
    image: np.ndarray,
    blur_method: str,
    kernel_size: int = 5
) -> tuple[str, np.ndarray]:
    """
    Blur an image with the specified blur method.
    :param image:
    :param blur_method:
    :param kernel_size:
    :return: Tuple of blurred image name and blurred image
    """

    if blur_method == "median":
        blurred_image = cv.medianBlur(image, kernel_size)
        return f"Median-{kernel_size}", blurred_image

    if blur_method == "gaussian":
        # Sigma is set to 0 so OpenCV works it out from the kernel size
        blurred_image = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return f"Gaussian-{kernel_size}", blurred_image

    if blur_method == "box":
        blurred_image = cv.blur(image, (kernel_size, kernel_size))
        return f"Box-{kernel_size}", blurred_image

    raise ValueError(f"Unknown blur method: {blur_method}")


def test_filtering(
    image_path: str,
    colour_space: str,
    channel: str,
    blur_methods: list[str],
    threshold_method: str,
    kernel_size: int = 5,
    threshold: int = 255,
    block_size: int = 101,
    c: int = -2,
    display: bool = False,
) -> None:
    """
    Test blurring/filtering methods on an image
    :param image_path: The base image
    :param colour_space: Colour space to use
    :param channel: Channel to use
    :param blur_methods: Blur methods to use
    :param threshold_method: Threshold method to use
    :param kernel_size:
    :param threshold:
    :param block_size:
    :param c:
    :param display:
    :return:
    """

    image = cv.imread(image_path)
    if image is None:
        print(f"Failed to load image at: {image_path}")
        return

    image_name = Path(image_path).stem
    output_dir = f"results/tests/filtering/{image_name}"

    # Blur the original image and then create a mask for the specified channel
    for blur_method in blur_methods:
        blur_name, blurred_image = get_blur(image, blur_method, kernel_size)
        channel_image = get_channels(blurred_image, colour_space)[channel]
        mask_name, mask = get_mask(channel_image, threshold_method, threshold, block_size, c)
        result = cv.bitwise_and(image, image, mask=mask)

        output_image(mask, output_dir, f"{blur_name}-{channel}-{mask_name}-Mask", display)
        output_image(result, output_dir, f"{blur_name}-{channel}-{mask_name}-Result", display)