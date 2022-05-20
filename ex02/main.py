# SCC0251 (2022/1)
# Assignment 2 - Image Ennhancement and Filtering
#
# Lucas Viana Vilela
# 10748409

from statistics import median

import imageio
import numpy as np


def norm(matrix: np.ndarray, new_max: float) -> np.ndarray:
    """
    Normalize a matrix.

    Parameters
    ----------
    matrix (np.ndarray): the matrix to be normalized
    new_max (float): the largest number in the normalized matrix

    Returns
    -------
    np.ndarray: the normalized matrix
    """
    max_matrix = np.max(matrix)
    min_matrix = np.min(matrix)

    # in case all elements are the same
    # (prevents dividing by zero)
    if min_matrix == max_matrix:
        matrix[:, :] = new_max
        return matrix

    # normalize the image so that min becomes 0 and max becomes `new_largest`
    return ((matrix - min_matrix) / (max_matrix - min_matrix)) * new_max


def limiarization(input_img: np.array, T0: float):
    n, m = input_img.shape
    T = T0

    while True:
        # binarize the image and calculate the average of each region
        avg_1 = np.mean(input_img[input_img > T])
        avg_2 = np.mean(input_img[input_img <= T])

        # update estimate treshold
        T0, T = T, np.mean((avg_1, avg_2))

        # if treshold is optimal
        if abs(T - T0) < 0.5:
            break

    img = np.zeros((n, m))
    img[input_img > T] = 1

    return img


def filtering_1D(input_img: np.array, n: int, weights: np.array):
    flat = np.pad(input_img.flatten(), n // 2, "wrap")
    arr = np.zeros(len(flat) - 2 * (n // 2))

    # loop through the array and calculate the new value for each position
    for i in range(len(arr)):
        np.dot(weights, flat[i : (i + n)])

    return arr.reshape(input_img.shape)


def filterig_2D_limiarization(input_img, n, weights, T0):
    img = np.zeros(input_img.shape)
    padded = np.pad(input_img, n // 2, "edge")

    # loop through each pixel of the image and calculate the new value
    for i in range(len(img)):
        for j in range(len(img)):
            img[i, j] = np.sum(padded[i : (i + n), j : (j + n)] * np.array(weights))

    return limiarization(img, T0)


def median_filter(input_image, n):
    img = np.zeros(input_image.shape)
    padded = np.pad(input_image, n // 2)

    # loop through each pixel and calculate the median for the respective window
    for i in range(len(img)):
        for j in range(len(img)):
            img[i, j] = median(padded[i : (i + n), j : (j + n)].reshape(-1))

    return img


if __name__ == "__main__":
    input_img_filename = input().strip()
    method = int(input())

    input_img = imageio.imread(input_img_filename)
    m, n = input_img.shape

    # read the correct methods' parameters and call it
    if method == 1:
        T0 = float(input())

        output_img = limiarization(input_img, T0)
    elif method == 2:
        n = int(input())
        weights = np.array([float(s) for s in input().split()])

        output_img = filtering_1D(input_img, n, weights)
    elif method == 3:
        n = int(input())
        weights = np.array([[float(s) for s in input().split()] for _ in range(n)])
        T0 = float(input())

        output_img = filterig_2D_limiarization(input_img, n, weights, T0)
    elif method == 4:
        n = int(input())

        output_img = median_filter(input_img, n)

    # normalize and convert
    output_img = norm(output_img, 255).astype(np.uint8)

    # output the comparison
    RMSE = np.sqrt(np.sum((input_img - output_img) ** 2) / (m * n))
    print(round(RMSE, 4))
