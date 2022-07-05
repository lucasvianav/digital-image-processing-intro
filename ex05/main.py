import enum
from math import ceil

import imageio
import numpy as np
from numpy.typing import NDArray


class MorphologyOperation(enum.Enum):
    """Types of morphology operation."""

    Opening = 1
    Closing = 2


def grayscale(img: NDArray) -> NDArray[np.number]:
    """
    Convert a colored image (RGB) to grayscale using the Luminance weights.

    Parameters
    ----------
    img: the image to be normalized

    Returns
    -------
    The grayscale image
    """
    grayscale_transform = [0.299, 0.587, 0.114]
    return img.dot(grayscale_transform)


def binarize(img: NDArray[np.number], threshold: float) -> NDArray[np.number]:
    """
    Convert an image to a binary image - all pixels with value below a threshold
    are black (0) and all above it are white (1).

    Parameters
    ----------
    img: the image to be converted.
    threshold: the threshold to binarize upon.

    Returns
    -------
    The binary image
    """
    binary_image = np.ones(img.shape).astype(np.uint8)
    binary_image[img < threshold] = 0
    return binary_image


def slice_img(mat: NDArray, i: int, j: int, size=3) -> NDArray:
    """
    Slice a matrix's square window in a certain point. Default to 3x3 window.

    Parameters
    ----------
    mat: the matrix to be sliced.
    i: the window's center element's row's index.
    j: the window's center element's column's index.
    size: the window's size.

    Returns
    -------
    The resulting slice/window.
    """
    quo_floor = size // 2
    quo_ceil = ceil(size / 2)
    return mat[i - quo_floor : i + quo_ceil, j - quo_floor : j + quo_ceil]


def bin_erosion(img: NDArray[np.number]) -> NDArray[np.uint8]:
    """
    Apply the erosion operation (mathematical morphology) to an image.

    Parameters
    ----------
    img: the image to be eroded.

    Returns
    -------
    The resulting image.
    """
    rows, cols = img.shape[:2]
    morphology = np.zeros(img.shape).astype(np.uint8)

    for i in np.arange(1, rows):
        for j in np.arange(1, cols):
            if slice_img(img, i, j).all():
                morphology[i, j] = 1

    return morphology


def bin_dilatation(img: NDArray[np.number]) -> NDArray[np.uint8]:
    """
    Apply the dilatation operation (mathematical morphology) to an image.

    Parameters
    ----------
    img: the image to be dilated.

    Returns
    -------
    The resulting image.
    """
    rows, cols = img.shape
    morphology = np.zeros(img.shape).astype(np.uint8)

    for i in np.arange(1, rows):
        for j in np.arange(1, cols):
            if slice_img(img, i, j).any():
                morphology[i, j] = 1

    return morphology


def bin_closing(img: NDArray[np.number]) -> NDArray[np.uint8]:
    """
    Apply the closing operation (mathematical morphology) to an image.

    Parameters
    ----------
    img: the image to be "closed".

    Returns
    -------
    The resulting image.
    """
    return bin_erosion(bin_dilatation(img))


def bin_opening(img: NDArray[np.number]) -> NDArray[np.uint8]:
    """
    Apply the opening operation (mathematical morphology) to an image.

    Parameters
    ----------
    img: the image to be "opened".

    Returns
    -------
    The resulting image.
    """
    return bin_dilatation(bin_erosion(img))


def extract_masks(
    img: NDArray[np.number], threshold: float, operation: MorphologyOperation
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    """
    Split an image in two masks considering a given threshold and morphological operation.

    Parameters
    ----------
    img: the target image.
    threshold: the threshold to binarize the image with.
    operation: the morphological operation to be executed in the binary image.

    Returns
    -------
    The image's sections result from applying the resulting post-morphological operation image as well as it's complement.
    """
    gray_img = grayscale(img).astype(np.uint8)
    bin_img = binarize(gray_img, threshold)
    morpholgy = (
        bin_opening(bin_img)
        if operation is MorphologyOperation.Opening
        else bin_closing(bin_img)
    )

    return (1 - morpholgy, morpholgy) * gray_img


def haralick_descriptors(pmatrix: NDArray[np.floating]) -> list[float]:
    """
    Split an image in two masks considering a given threshold and morphological operation.

    Parameters
    ----------
    pmatrix: the GLCM probability matrix.

    Returns
    -------
    The list of haralick decriptors in order: auto_correlation, contrast,
    dissimilarity, energy, entropy, homogeneity, inverse_difference and
    maximum_probability.
    """
    rows, cols = pmatrix.shape
    I, J = np.ogrid[:rows, :cols]

    haralick_descriptors: list[float] = [
        (pmatrix * (I * J)).sum(),  # auto_correlation
        ((I - J) ** 2 * pmatrix).sum(),  # contrast
        (np.abs(I - J) * pmatrix).sum(),  # dissimilarity
        (pmatrix**2).sum(),  # energy
        -((mat := pmatrix[pmatrix > 0]) * np.log(mat)).sum(),  # entropy
        (pmatrix / (1 + (I - J) ** 2)).sum(),  # homogeneity
        (pmatrix / (1 + np.abs(I - J))).sum(),  # inverse_difference
        np.max(pmatrix),  # maximum_probability
    ]

    return haralick_descriptors


def probabilities_GLCM(
    img: NDArray[np.integer], Q: tuple[int, int]
) -> NDArray[np.floating]:
    """
    Calculate the GLCM probability matrix for a given image and distance vector.

    Parameters
    ----------
    img: the target image.
    Q: the distance 2D vector between reference pixel and neighbor pixel.

    Returns
    -------
    The GLCM probability matrix.
    """
    co_ocurrence_matrix = np.zeros(((size := img.max()) + 1, size + 1))
    rows, cols = img.shape
    dx, dy = Q

    for i in np.arange(1, rows - 1):
        for j in np.arange(1, cols - 1):
            reference = img[i, j]
            neighbor = img[i + dx, j + dy]
            co_ocurrence_matrix[reference, neighbor] += 1

    return co_ocurrence_matrix / co_ocurrence_matrix.sum()


def euclidian_distance(p: NDArray[np.floating], q: NDArray[np.floating]) -> float:
    """Calculate the euclidian distance between two given R^N vectors of same shape."""
    return ((p - q) ** 2).sum() / p.shape[0]


if __name__ == "__main__":
    # inputs
    query = int(input().strip())
    Q = tuple(int(c) for c in input().strip().split())
    operation = MorphologyOperation(int(input().strip()))
    binarization_threshold = float(input().strip())
    n_images = int(input().strip())
    image_dataset = [input().strip() for _ in range(n_images)]

    imgs_descriptors = []  # all descriptors for each image
    for filename in image_dataset:
        img = imageio.imread(filename)
        mask1, mask2 = extract_masks(img, binarization_threshold, operation)
        descriptors1 = haralick_descriptors(probabilities_GLCM(mask1, Q))
        descriptors2 = haralick_descriptors(probabilities_GLCM(mask2, Q))

        # save descriptors for current image
        imgs_descriptors.append(np.concatenate((descriptors1, descriptors2), axis=None))

    # distances between each image and the queries image
    distances = np.array(
        [euclidian_distance(curr, imgs_descriptors[query]) for curr in imgs_descriptors]
    )

    # images sorted from lowest to highest distances (highest to lowest similarities)
    ranking = sorted(
        [(image_dataset[i], distances[i]) for i in range(n_images)],
        key=lambda e: e[1],
    )

    # print output
    print(f"Query: {image_dataset[query]}", "Ranking:", sep="\n")
    for i in range(len(ranking)):
        print(f"({i}) {ranking[i][0]}")
