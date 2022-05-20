# SCC0251 (2022/1)
# Assignment 1 - Image Generation
#
# Lucas Viana Vilela
# 10748409

from random import randint, random, seed

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

    # normalize the image so that min becomes 0 and max becomes `new_largest`
    return ((matrix - min_matrix) / (max_matrix - min_matrix)) * new_max


def randomwalk(matrix: np.ndarray) -> np.ndarray:
    """
    Populate a matrix using the randomwalk algorithm (described in the assingment).

    Parameters
    ----------
    matrix (np.ndarray): the matrix to be populated

    Returns
    -------
    np.ndarray: the populated matrix
    """
    rows, cols = matrix.shape

    x = y = 0
    for _ in range(rows * cols + 1):
        matrix[x, y] = 1

        dx, dy = randint(-1, 1), randint(-1, 1)
        x = (x + dx) % rows
        y = (y + dy) % cols

    return matrix


# methods equivalent to the functions described
# in the assignment to create the image
functions = {
    1: lambda x, y: x * y + 2 * y,
    2: lambda x, y: abs(np.cos(x / Q) + 2 * np.sin(y / Q)),
    3: lambda x, y: abs(3 * (x / Q) - np.cbrt(y / Q)),
    4: lambda x, y: random(),
}

if __name__ == "__main__":
    # parameter input
    filename = input().strip()
    C = int(input())
    function = int(input())
    Q = int(input())
    N = int(input())  # <= C
    B = int(input())  # in [1, 8]
    S = int(input())

    # set random seed
    seed(S)

    f = np.zeros((C, C), dtype=float)

    # populate f (create the image)
    if function == 5:
        f = randomwalk(f)
    else:
        for x in range(C):
            for y in range(C):
                f[x, y] = functions[function](x, y)

    f = norm(f, 2**16 - 1)
    g = np.zeros((N, N), dtype=float)
    step = C // N

    # downsampling
    for x in range(N):
        for y in range(N):
            g[x, y] = f[x * step, y * step]

    # normalize, convert and bitshift
    g = norm(g, 255).astype(np.uint8) >> (8 - B)

    # reference image
    R = np.load(filename)

    # output the Root Squared Error (RSE)
    root_squared_error = np.sqrt(np.sum((g - R) ** 2))
    print(round(root_squared_error, 4))
