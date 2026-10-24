import cv2 as cv
import glob
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT/"Data"/"Samples"/"Images"
TESTS_DIR = PROJECT_ROOT/"Data"/"Tests"/"Channels"
IMAGES = {}

# Store all image paths in a dict keyed by filename
for path in glob.glob(f"{IMAGE_DIR}/*.png"):
    name = os.path.splitext(os.path.basename(path))[0]
    IMAGES[name] = path


def output_image_properties(image_path: str) -> None:
    img = cv.imread(image_path)
    print(
        f"Properties of {image_path}:\n",
        f"Number of pixels: {img.size}\n",
        f"Datatype: {img.dtype}\n",
        f"Shape (rows, columns, channels) = {img.shape}"
    )


def test_different_channels(
    image_path: str,
    rgb: bool = False,
    hsv: bool = False,
    lab: bool = False,
    display: bool = True
) -> None:
    """
    Convert input image into different color channels and write to tests dir.
    :param image_path: Path of image to be converted.
    :param rgb: Whether to convert RGB channels.
    :param hsv: Whether to convert HSV channels.
    :param lab: Whether to convert LAB channels.
    :param display: Whether to display images in windows.
    :return: None
    """

    img = cv.imread(image_path)
    if img is None:
        print("Failed to load image")
        return

    def _output(name: str, channel, display: bool) -> None:
        if display:
            cv.imshow(name, channel)
        else:
            cv.imwrite(str(TESTS_DIR / f"{name}.png"), channel)

    def _split_rgb(image: img, display: bool) -> None:
        img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        r, g, b = cv.split(img_rgb)
        _output("RGB", img_rgb, display)
        _output("RGB-Red", r, display)
        _output("RGB-Green", g, display)
        _output("RGB-Blue", b, display)

    def _split_hsv(image: img, display: bool) -> None:
        img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(img_hsv)
        _output("HSV-Hue", h, display)
        _output("HSV-Saturation", s, display)
        _output("HSV-Value", v, display)

    def _split_lab(image: img, display: bool) -> None:
        img_lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(img_lab)
        _output("LAB-Brightness (L)", l, display)
        _output("LAB-Red-Green (A)", a, display)
        _output("LAB-Blue-Yellow (B)", b, display)

    if rgb:
        _split_rgb(img, display)
    if lab:
        _split_lab(img, display)
    if hsv:
        _split_hsv(img, display)



if __name__ == "__main__":
    print("=====Output Start=====")

    path = IMAGES["splat_360"]

    # test_different_channels(path, display=False)

    cv.waitKey(0)
    cv.destroyAllWindows()