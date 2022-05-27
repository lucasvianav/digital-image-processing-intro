# SCC0251 (2022/1)
# Assignment 4 - Image Restoration
#
# Lucas Viana Vilela
# 10748409

import imageio
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from numpy.typing import NDArray


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


def gaussian_blur(k, sigma) -> NDArray:
    """
    Create a Gaussian blur degradation matrix.

    Parameters
    ----------
    k (int): lateral size of the kernel/filter, default 5
    sigma (float): Gaussian distribution's standard deviation.

    Returns
    -------
    NDArray: the filter matrix.
    """
    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filter = np.exp(-(1 / 2) * (np.square(x) + np.square(y)) / np.square(sigma))
    return filter / np.sum(filter)


def motion_blur(shape: tuple[int, int], angle: float, num_pixel_dist=20) -> NDArray:
    """
    Create a motion blur (PSF) degradation matrix.

    Parameters
    ----------
    shape (tuple[int, int]): desired number os rows and columns.
    angle (float): motion blur's angle in degrees - valid interval [0, 360).
    num_pixel_dist (int): motion blur's distance in pixels (minimum 0, default 20).

    Returns
    -------
    NDArray: the filter matrix.
    """
    if not 0 <= angle < 360:
        raise ValueError(f"angle = {angle} is invalid, it should be 0 <= angle < 360.")
    elif num_pixel_dist < 0:
        raise ValueError("num_pixel_dist must be >= 0.")

    psf = np.zeros(shape)
    angle_radians = angle / 180 * np.pi

    center = [(dim - 1) // 2 for dim in shape]
    phase = [np.cos(angle_radians), np.sin(angle_radians)]

    for i in range(num_pixel_dist):
        offset_x = center[0] - round(i * phase[0])
        offset_y = center[1] - round(i * phase[1])
        psf[offset_x, offset_y] = 1

    return psf / psf.sum()


def least_squares(g: NDArray, h: NDArray, gamma: float) -> NDArray[np.uint8]:
    """
    Constrained least-squares method for digital image restoration.

    Parameters
    ----------
    g (NDArray): image to be restored.
    h (NDArray): gaussian degradation (blurring) matrix.
    gamma (float): regularization parameter - must be in the interval [0, 1).

    Returns
    -------
    NDArray[uint8]: the restored image.
    """
    if not 0 <= gamma < 1:
        raise ValueError(f"gamma = {gamma} is invalid, it should be 0 <= gamma < 1.")

    # laplacian operator
    p = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    # fourier transforms
    P: NDArray = fftshift(fft2(p, g.shape))
    H: NDArray = fftshift(fft2(h, g.shape))
    G: NDArray = fftshift(fft2(g))

    # apply the CLSQ method and transform back to space domain
    F_hat = (H.conj() / (np.abs(H) ** 2 + gamma * np.abs(P) ** 2)) * G
    f_hat = ifft2(ifftshift(F_hat))

    # clip and convert output to uint8
    return np.clip(np.real(f_hat), 0, 255).astype(np.uint8)


def richardson_lucy(g: NDArray, psf: NDArray, n_steps: int):
    """
    Richardsn-Lucy deconvolution for digital image restration (from motion blur).

    Parameters
    ----------
    g (NDArray): image to be restored.
    psf (NDArray): point spread function (motion blur) matrix.
    n_steps (int): number of iteration steps to be performed.

    Returns
    -------
    NDArray[uint8]: the restored image.
    """
    # initial image
    r = np.full(shape=g.shape, fill_value=1, dtype="float64")

    # fourier transforms
    R = fft2(r)
    PSF = fft2(psf)

    for _ in range(n_steps):
        tmp = fft2(g / ifft2(R * PSF)) * np.flip(PSF)
        r *= np.real(ifft2(tmp))
        R = fft2(r)

    return np.clip(r, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    input_img_filename = input().strip()
    method = int(input().strip())

    input_img = imageio.imread(input_img_filename)

    if method == 1:  # CLSQ
        # read input
        k = int(input())
        sigma = float(input())
        gamma = float(input())

        # generate the degradation "function"
        h = gaussian_blur(k, sigma)
        H: NDArray = fftshift(fft2(h, input_img.shape))

        # generate the the blurred image
        F: NDArray = fftshift(fft2(input_img))
        G = F * H
        g = np.real(ifft2(ifftshift(G)))

        restored_img = least_squares(g, h, gamma)
    else:  # RL
        # read input
        angle = int(input())
        n_steps = int(input())

        # generate the point spread "function"
        psf = motion_blur(input_img.shape, angle)
        PSF: NDArray = fftshift(fft2(psf))

        # generate the motion-blurred image
        F: NDArray = fftshift(fft2(input_img))
        G = F * PSF
        g = np.real(ifft2(ifftshift(G)))

        restored_img = richardson_lucy(g, psf, n_steps)

    # output the comparison between the original image and the restred one
    error = rmse(input_img, restored_img)
    print(round(error, 4))
