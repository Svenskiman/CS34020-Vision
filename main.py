import cv2 as cv
import numpy as np
import matplotlib

from tests.test_channels import test_channels
from tests.test_thresholding import test_thresholding
from tests.test_filtering import (
    test_filtering,
    get_blur,
    examine_image_noise,
)
from tests.test_contours import test_contours


if __name__ == "__main__":

    print(
        f"OpenCV version: {cv.__version__},\n"
        f"NumPy version: {np.__version__},\n"
        f"Matplotlib version: {matplotlib.__version__},\n"
    )

    display = False
    sample = 320
    path = f"Data/Samples/Images/splat_{sample}.png"
    # mask = "results/tests/filtering/splat_390/Gaussian-33-LAB-Red-Green (A)-Binary-160-Mask.png"

    # test_channels(
    #     image_path = path,
    #     colour_spaces = ["RGB", "HSV", "LAB"],
    #     display = display,
    # )

    # test_thresholding(
    #     image_path=path,
    #     colour_space="LAB",
    #     channel="LAB-Red-Green (A)",
    #     threshold_methods=["binary", "otsu", "adaptive"],
    #     threshold=160,
    #     block_size=101,
    #     c=-2,
    #     display=display,
    # )

    test_filtering(
        image_path=path,
        colour_space="LAB",
        channel="LAB-Red-Green (A)",
        # blur_methods=["gaussian", "median", "box"],
        blur_methods=["gaussian"],
        threshold_method="binary",
        kernel_size=33,
        threshold=160,
        plot_intensity_profile=True,
    )


    gaussian_mask = f"results/tests/filtering/splat_{sample}/Gaussian-33-LAB-Red-Green (A)-Binary-160-Mask.png"

    test_contours(
        image_path=path,
        mask_path=gaussian_mask,
        min_area=100,
        display=display,
    )

    if display:
        cv.waitKey(0)
        cv.destroyAllWindows()