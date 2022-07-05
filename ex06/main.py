# SCC0251 (2022/1)
# Assignment 6 - Color Image Processing and Segmentation
#
# Lucas Viana Vilela
# 10748409

import random
from enum import Enum

import imageio
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def plot(imgs: list[dict], n_rows: int, n_cols: int, axis=True) -> None:
    """
    Plot a list of images using MatPlotLib.

    Parameters
    ----------
    imgs ({ "title": str, "images": NDArray }): images to be plotted
    n_rows (int): number of rows in the grid
    n_cols (int): number of columns in the grid
    axis (boolean): should the axis be shown in the subplots
    """
    actual_n_rows = len(imgs) // n_cols
    n_cols_last_row = len(imgs) - (actual_n_rows - 1) * n_cols

    for i in range(1, actual_n_rows + 1):
        for j in range(1, n_cols_last_row + 1):
            img = imgs[(index := (i - 1) * n_cols + j) - 1]
            plt.subplot(int(f"{n_rows}{n_cols}{index}"))
            plt.axis("on" if axis else "off")
            plt.title(img["title"])
            plt.imshow(
                img["image"].astype(np.uint8),
                cmap=img["cmap"] if "cmap" in img else "gray",
            )
    plt.show()


class PixelAttributeOption(Enum):
    """Option for pixel attributes."""

    RGB = 1
    RGBxy = 2
    Luminance = 3
    LuminanceXY = 4


def norm(img: NDArray, new_max=255) -> NDArray:
    """
    Normalize a single channel image (e.g. grayscale).

    Parameters
    ----------
    img: the image to be normalized
    new_max: the largest value in the normalized image, default 255

    Returns
    -------
    The normalized image
    """
    max_value, min_value = img.max(), img.min()
    return ((img - min_value) / (max_value - min_value)) * new_max


def multi_channel_norm(img: NDArray, new_max=255) -> NDArray:
    """
    Normalize a single channel image (e.g. grayscale).

    Parameters
    ----------
    img: the image to be normalized
    new_max: the largest value in the normalized image, default 255

    Returns
    -------
    The normalized image
    """
    normalized = img.copy()
    for i in np.arange(img.shape[2]):
        img[:, :, i] = norm(img[:, :, i], new_max)
    return normalized


def rmse(image1: NDArray[np.uint8], image2: NDArray[np.uint8]) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between two single channel images (e.g.: grayscale).

    This method is not order-sensitive and assumes that both matrices have the same shape.

    Returns
    -------
    float: the error between the two matrices
    """
    m, n = image1.shape
    return np.sqrt(np.sum((image1 - image2) ** 2) / (m * n))


def multi_channel_rmse(image1: NDArray[np.uint8], image2: NDArray[np.uint8]) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between two multi channel images (e.g.: RGB).

    This method is not order-sensitive and assumes that both matrices have the same shape.

    Returns
    -------
    float: the error between the two matrices
    """
    acc = 0
    for i in np.arange(n_channels := image1.shape[2]):
        acc += rmse(image1[:, :, i], image2[:, :, i])
    return acc / n_channels


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


def reshape(matrix: NDArray) -> NDArray:
    n_channels = matrix.shape[2] if len(matrix.shape) == 3 else 1
    m, n = matrix.shape[:2]
    return matrix.reshape((m * n, n_channels))


def coordinates_matrix(m, n):
    x_coords = np.tile(np.arange(m).reshape((m, 1)), (1, n)).reshape((m * n, 1))
    y_coords = np.tile(np.arange(n).reshape((1, n)), (m, 1)).reshape((m * n, 1))
    return np.concatenate((x_coords, y_coords), axis=1)


def distance(p: NDArray[np.float32], q: NDArray[np.float32]) -> float:
    """Calculate the euclidian distance between two given R^N vectors of same shape."""
    return ((p - q) ** 2).sum() / p.shape[0]


def parse_dataset_from_img(img: NDArray, option: PixelAttributeOption) -> NDArray:
    evaluate = {
        PixelAttributeOption.RGB: lambda img: reshape(img),
        PixelAttributeOption.RGBxy: lambda img: np.concatenate(
            (reshape(img), coordinates_matrix(*img.shape[:2])), axis=1
        ),
        PixelAttributeOption.Luminance: lambda img: reshape(grayscale(img)),
        PixelAttributeOption.LuminanceXY: lambda img: np.concatenate(
            (reshape(grayscale(img)), coordinates_matrix(*img.shape[:2])), axis=1
        ),
    }[option]

    return evaluate(img)


def parse_img_from_features(
    features: NDArray, option: PixelAttributeOption
) -> NDArray:
    evaluate = {
        PixelAttributeOption.RGB: lambda R, G, B: [R, G, B],
        PixelAttributeOption.RGBxy: lambda R, G, B, x, y: [R, G, B],
        PixelAttributeOption.Luminance: lambda l: l,
        PixelAttributeOption.LuminanceXY: lambda l, x, y: l,
    }[option]

    multi_channel = {
        PixelAttributeOption.RGB: True,
        PixelAttributeOption.RGBxy: True,
        PixelAttributeOption.Luminance: False,
        PixelAttributeOption.LuminanceXY: False,
    }[option]

    m, n = features.shape[:2]
    img = np.zeros((m, n, 3) if multi_channel else (m, n))

    for i in np.arange(m):
        for j in np.arange(n):
            img[i, j] = evaluate(*features[i, j]) if multi_channel else evaluate(features[i, j])

    return img.astype(np.float32)


def k_means(k: int, img: NDArray, n_iterations: int):
    m, n = img.shape[:2]
    dataset = parse_dataset_from_img(img, attributes)
    clusters = np.zeros(m * n).astype(np.uint32)

    # initialize centroids
    initial_centroids_indexes = np.sort(random.sample(range(m * n), k))
    centroids = np.array([dataset[i] for i in initial_centroids_indexes])

    for _ in np.arange(n_iterations):
        # assign each pixel to closest centroid
        for i in np.arange(m * n):
            distances = [distance(centroid, dataset[i]) for centroid in centroids]
            clusters[i] = np.argmin(distances)

        # recalculate centroids
        for i in np.arange(k):
            centroids[i] = (
                np.mean(dataset[slice])
                if (slice := clusters == i).any()
                else centroids[i]
            )

    return centroids[clusters].reshape((m, n))


if __name__ == "__main__":
    input_img_filename = input().strip()
    ref_img_filename = input().strip()
    attributes = PixelAttributeOption(int(input().strip()))
    n_clusters = int(input().strip())
    n_iterations = int(input().strip())
    seed = int(input().strip())

    random.seed(seed)

    input_img = imageio.imread(input_img_filename).astype(np.float32)
    ref_img = imageio.imread(ref_img_filename)
    multi_channel = len(ref_img.shape) == 3

    clusters = k_means(n_clusters, input_img, n_iterations)
    output_img = parse_img_from_features(clusters, attributes)
    output_img = multi_channel_norm(output_img) if multi_channel else norm(output_img)

    print(
        multi_channel_rmse(ref_img, output_img)
        if multi_channel
        else rmse(ref_img, output_img)
    )

    plot(
        [
            {"title": "Original", "image": input_img, "cmap": "viridis"},
            {"title": "Theirs", "image": ref_img, "cmap": "viridis"},
            {"title": "Mine", "image": output_img, "cmap": "viridis"},
        ],
        1,
        3,
    )
