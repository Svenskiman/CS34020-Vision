import cv2 as cv
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

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


def plot_intensity_profiles(
    images_to_plot: dict[str, np.ndarray],
    channel: str,
    output_dir: str,
    row: int = 50,
) -> None:
    """
    Plot intensity profiles for each image
    :param images_to_plot: Un-blurred and blurred images
    :param channel: Name of channel used
    :param output_dir:
    :param row:
    :return:
    """
    name = f"Intensity-Profile-Row-{row}-{channel}"

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{name}.png"

    fig, ax = plt.subplots()
    for label in images_to_plot:
        ax.plot(images_to_plot[label][row, :], label=label)
    ax.set(ylim=(0, 255), xlabel="Column", ylabel="Intensity", title=name)
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Plot {name} saved at: {output_path}")


def examine_image_noise(
    image_to_examine: np.ndarray,
) -> None:
    """
    https://www.geeksforgeeks.org/python/opencv-python-program-analyze-image-using-histogram/
    Plot histogram of an image to check noise type.
    :param image_to_examine:
    :return: None
    """
    img = cv.imread(image_to_examine, cv.IMREAD_GRAYSCALE)

    # Plot image and compare with mask to find coordinates of interest
    # plt.imshow(img, cmap="gray")
    # plt.show()

    """I used splat_360.png for testing"""
    # Check weird white blobs that show in mask that aren't strawberries
    part = img[300:340, 220:260]
    plt.hist(part.ravel(), 256, [0, 256], label="White blob (not strawberry)")

    # Check strawberry
    part = img[550:590, 540:580]
    plt.hist(part.ravel(), 256, [0, 256], label="Strawberry")

    plt.legend()
    plt.show()


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
    plot_intensity_profile: bool = False,
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
    :param plot_intensity_profile: Plot intensity profiles if true
    :return:
    """

    image = cv.imread(image_path)
    if image is None:
        print(f"Failed to load image at: {image_path}")
        return

    image_name = Path(image_path).stem
    output_dir = f"results/tests/filtering/{image_name}"
    plot_dir = f"{output_dir}/plots"

    images_to_plot = {}
    if plot_intensity_profile:
        # Unblurred image to plot against
        images_to_plot["None"] = get_channels(image, colour_space)[channel]

    # Blur the original image and then create a mask for the specified channel
    for blur_method in blur_methods:
        blur_name, blurred_image = get_blur(image, blur_method, kernel_size)
        channel_image = get_channels(blurred_image, colour_space)[channel]
        if plot_intensity_profile:
            images_to_plot[blur_name] = channel_image

        mask_name, mask = get_mask(channel_image, threshold_method, threshold, block_size, c)
        result = cv.bitwise_and(image, image, mask=mask)

        output_image(mask, output_dir, f"{blur_name}-{channel}-{mask_name}-Mask", display)
        output_image(result, output_dir, f"{blur_name}-{channel}-{mask_name}-Result", display)

    if plot_intensity_profile:
        plot_intensity_profiles(images_to_plot, channel, plot_dir)