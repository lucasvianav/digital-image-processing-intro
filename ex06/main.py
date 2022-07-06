# SCC0251 (2022/1)
# Assignment 6 - Color Image Processing and Segmentation
#
# Lucas Viana Vilela
# 10748409

import random
from enum import Enum

import imageio
import numpy as np
from numpy.typing import NDArray


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
    return (((img - min_value) / (max_value - min_value)) * new_max).astype(np.float32)


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
    for i in range(img.shape[2]):
        img[:, :, i] = norm(img[:, :, i], new_max)
    return normalized


def rmse(image1: NDArray[np.float32], image2: NDArray[np.float32]) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between two single channel images (e.g.: grayscale).

    This method is not order-sensitive and assumes that both matrices have the same shape.

    Returns
    -------
    float: the error between the two matrices
    """
    m, n = image1.shape
    return np.sqrt(np.sum((image1 - image2) ** 2) / (m * n))


def multi_channel_rmse(image1: NDArray, image2: NDArray) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between two multi channel images (e.g.: RGB).

    This method is not order-sensitive and assumes that both matrices have the same shape.

    Returns
    -------
    float: the error between the two matrices
    """
    acc = 0
    for i in range(n_channels := image1.shape[2]):
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
    """Reshape a "matrix" in which each element may be a numpy array to a "row" array, maintaining each element as it is."""
    n_channels = matrix.shape[2] if len(matrix.shape) == 3 else 1
    m, n = matrix.shape[:2]
    return matrix.reshape((m * n, n_channels))


def coordinates_matrix(m: int, n: int) -> NDArray:
    """
    Generate a matrix in which each element is an array containing that element's row and column index.

    Parameters
    ----------
    m: number of rows in the matrix
    n: number of columns in the matrix

    Returns
    -------
    The matrix containing the coordinates
    """
    x_coords = np.tile(np.arange(m).reshape((m, 1)), (1, n)).reshape((m * n, 1))
    y_coords = np.tile(np.arange(n).reshape((1, n)), (m, 1)).reshape((m * n, 1))
    return np.concatenate((x_coords, y_coords), axis=1)


def euclidian_dist(p: NDArray[np.float32], q: NDArray[np.float32], axis=1) -> NDArray:
    """Calculate the euclidian distance between two given R^N vectors of same shape."""
    return ((p - q) ** 2).sum(axis=axis) / p.shape[0]


def parse_dataset_from_img(img: NDArray, option: PixelAttributeOption) -> NDArray:
    """Convert an image to a dataset with the chosen attributes."""
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


def parse_img_from_features(features: NDArray, option: PixelAttributeOption) -> NDArray:
    """Convert a matrix of features (according to the chosen attributes) into an RGB or grayscale image."""
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

    for i in range(m):
        for j in range(n):
            img[i, j] = (
                evaluate(*features[i, j])
                if option != PixelAttributeOption.Luminance
                else evaluate(features[i, j])
            )

    return img.astype(np.float32)


def k_means(k: int, img: NDArray, n_iterations: int, seed: int) -> NDArray:
    """
    Partition an image to a given number of cluster.

    Parameters
    ----------
    k: number of clusters
    img: target image
    n_iterations: maximum number of iterations to be executed
    seed: seed to use in random generations

    Returns
    -------
    The clusterized image
    """
    m, n = img.shape[:2]
    clusters = np.zeros(m * n).astype(np.float32)
    features = parse_dataset_from_img(img, attributes)
    _, n_attributes = features.shape

    # initialize centroids
    random.seed(seed)
    centroids = features[random.sample(range(m * n), k)]

    for _ in range(n_iterations):
        # calculate distances between each pixel and each centroid
        distances = np.empty((m * n, 0))
        for centroid in centroids:
            centroid_vec = np.full((m * n, n_attributes), centroid)
            curr_distances = euclidian_dist(features, centroid_vec).reshape((m * n, 1))
            distances = np.append(distances, curr_distances, axis=1)

        # assign each pixel to closest centroid
        clusters = distances.argmin(axis=1)

        # recalculate centroids
        prev_centroids = centroids.copy()
        for i in range(k):
            centroids[i] = np.mean(features[clusters == i], axis=0)

        # check convergence
        if (prev_centroids == centroids).all():
            break

    # apply the clusters values to each pixel
    # and reshape to match the image's shape
    output = centroids[clusters].reshape(
        (*img.shape[:2], centroids.shape[1])
        if len(centroids.shape) == 2
        else img.shape[:2]
    )
    return parse_img_from_features(output, attributes)


if __name__ == "__main__":
    # read input
    input_img_filename = input().strip()
    ref_img_filename = input().strip()
    attributes = PixelAttributeOption(int(input().strip()))
    n_clusters = int(input().strip())
    n_iterations = int(input().strip())
    seed = int(input().strip())

    # setup
    input_img = imageio.imread(input_img_filename).astype(np.float32)
    ref_img = imageio.imread(ref_img_filename).astype(np.float32)
    multi_channel = len(ref_img.shape) == 3

    # apply k-means and normalize the output
    clusterized_img = k_means(n_clusters, input_img, n_iterations, seed)
    output_img = (
        multi_channel_norm(clusterized_img) if multi_channel else norm(clusterized_img)
    )

    # output the RMSE
    root_squared_error = (
        multi_channel_rmse(ref_img, output_img)
        if multi_channel
        else rmse(ref_img, output_img)
    )
    print(round(root_squared_error, 4))
