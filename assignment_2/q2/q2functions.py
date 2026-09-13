import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from PIL import Image

def nearest_neighbor_resize(image, scale_factor):

    old_height, old_width = image.shape

    # Calculate dimensions of the new image
    new_height = old_height // scale_factor
    new_width = old_width // scale_factor

    # Create empty output image
    resized = np.zeros((new_height, new_width), dtype=image.dtype)

    # nearest-neighbor sampling
    for y in range(new_height):
        for x in range(new_width):

            # Corresponding pixel in original image
            old_y = y * scale_factor
            old_x = x * scale_factor

            # Copy the pixel
            resized[y, x] = image[old_y, old_x]

    return resized

def ideal_lowpass_filter(shape, scale_factor):

    height, width = shape

    # New Nyquist frequency
    cutoff = 1 / (2 * scale_factor)

    # Frequency coordinates
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))

    # Create 2D frequency grids
    FX, FY = np.meshgrid(fx, fy)

    # Rectangular ideal LPF
    H = ((np.abs(FX) <= cutoff) & (np.abs(FY) <= cutoff))

    return H.astype(float)

def separable_butterworth_filter(shape, scale_factor, order):

    height, width = shape

    #Nyquist frequency
    cutoff = 1 / (2 * scale_factor)

    # Frequency coordinates
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))

    # 1-D Butterworth filters
    Hx = 1 / (1 + (np.abs(fx) / cutoff) ** (2 * order))

    Hy = 1 / (1 + (np.abs(fy) / cutoff) ** (2 * order))

    # Separable 2-D filter
    H = np.outer(Hy, Hx)

    return H

def create_notch_filter(shape, notch_points, radius):

    height, width = shape

    # Start by keeping every frequency
    H = np.ones((height, width))

    y = np.arange(height)
    x = np.arange(width)

    X, Y = np.meshgrid(x, y)

    # Create a zero-valued circular notch
    # around every selected frequency
    for point_y, point_x in notch_points:

        distance = np.sqrt((X - point_x) ** 2 + (Y - point_y) ** 2)
        H[distance <= radius] = 0

    return H

# 4. Optimum-notch weighting function w(x,y), computed over a local
#    (2a+1) x (2b+1) neighbourhood centred at every pixel:
#
#         mean(g*eta) - mean(g)*mean(eta)
#    w = ----------------------------------
#              mean(eta^2) - mean(eta)^2
#
#    All the mean terms are LOCAL averages (box filter of size
#    `win`), computed efficiently with scipy.ndimage.uniform_filter
#    instead of a manual double loop.

def optimum_notch_restore(g, eta, win):
    mean_g = uniform_filter(g, size=win, mode="reflect")
    mean_eta = uniform_filter(eta, size=win, mode="reflect")
    mean_g_eta = uniform_filter(g * eta, size=win, mode="reflect")
    mean_eta2 = uniform_filter(eta ** 2, size=win, mode="reflect")

    numerator = mean_g_eta - mean_g * mean_eta
    denominator = mean_eta2 - mean_eta ** 2

    eps = 1e-8  # avoid division by ~0 in flat regions of eta
    w = numerator / (denominator + eps)

    modulated_noise = w * eta
    restored = g - modulated_noise
    return w, modulated_noise, restored