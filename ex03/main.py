# SCC0251 (2022/1)
# Assignment 3 - Filtering in Spatial and Frequency Domain
#
# Lucas Viana Vilela
# 10748409

import imageio
import numpy as np
from numpy.typing import NDArray


def norm(matrix: NDArray[np.number], new_max: float) -> NDArray[np.number]:
    """
    Normalize a matrix.

    Parameters
    ----------
    matrix (NDArray[np.number]): the matrix to be normalized
    new_max (float): the largest number in the normalized matrix

    Returns
    -------
    NDArray[np.number]: the normalized matrix
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


def rmse(image1: NDArray[np.uint8], image2: NDArray[np.uint8]) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between two images (matrices).

    This method is not order-sensitive and assumes that both matrices have the same shape.

    Returns
    -------
    float: the error between the two matrices
    """
    m, n = image1.shape
    return np.sqrt(np.sum((image1 - image2) ** 2) / (m * n))


if __name__ == "__main__":
    # read inputs
    input_img_filename = input().strip()
    filter_img_filename = input().strip()
    reference_img_filename = input().strip()

    # read images + normalize filter
    input_img = imageio.imread(input_img_filename)
    filter_img = norm(imageio.imread(filter_img_filename), 1)
    reference_img = imageio.imread(reference_img_filename)

    # transform img into the frequency domain and shift it to the center
    input_img_frequency = np.fft.fftshift(np.fft.fft2(input_img))
    # filter the shifted image
    filtered_img_frequency = np.multiply(input_img_frequency, filter_img)
    # shift the image back and transform it to the space domain
    filtered_img_space = np.fft.ifft2(np.fft.ifftshift(filtered_img_frequency))

    # normalize and convert
    output_img = norm(np.real(filtered_img_space), 255).astype(np.uint8)

    # output the comparison between the reference image and the generated one
    error = rmse(reference_img, output_img)
    print(round(error, 4))
