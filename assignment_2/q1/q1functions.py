import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import cmath

def fft(image_arr):
    N = len(image_arr)

    #base case
    if(N == 1):
        return image_arr

    even = image_arr[0::2]
    odd = image_arr[1::2]

    even_fft = fft(even)
    odd_fft = fft(odd)

    output = [0]*N 

    for k in range (N//2):

        angle = -2*math.pi*k/N
        mul_factor = cmath.exp(1j*angle)

        output[k] = even_fft[k] + mul_factor*odd_fft[k]
        output[k + N//2] = even_fft[k] - mul_factor*odd_fft[k]

    return output

#2D FFT
def fft2(matrix):
    height = len(matrix)
    width = len(matrix[0])

    #FFT row
    row_fft = []
    for row in matrix:
        transformed_row = fft(row)
        row_fft.append(transformed_row)

    #output matrix
    result = []
    for i in range(height):
        result.append([0]*width)

    #FFT Col
    for col in range(width):
        column = []
        for row in range(height):
            column.append(row_fft[row][col])
        
        transformed_column = fft(column)

        for row in range(height):
            result[row][col] = transformed_column[row]

    return result

#Inverse FFT using FFT
def ifft(matrix):
    H = len(matrix)
    W = len(matrix[0])

    return np.conjugate(fft2(np.conjugate(matrix)))/H*W

#Padding to 2048x2048
def padding(img_matrix):
    target_height = 2048
    target_width = 2048

    pad_bottoom = target_height - img_matrix.shape[0]
    pad_right = target_width - img_matrix.shape[1]

    padded_matrix = np.pad(img_matrix,((0,pad_bottoom),(0,pad_right)), mode ='constant', constant_values = 0)

    return padded_matrix

def padding2(img_matrix):
    target_height = 256
    target_width = 256

    pad_bottoom = target_height - img_matrix.shape[0]
    pad_right = target_width - img_matrix.shape[1]

    padded_matrix = np.pad(img_matrix,((0,pad_bottoom),(0,pad_right)), mode ='constant', constant_values = 0)

    return padded_matrix

def create_disc_kernel(radius):

    # Create an odd-sized coordinate grid spanning from -radius to +radius
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    
    # Generate a boolean mask where distance from the center is <= radius
    mask = (x**2 + y**2) <= radius**2
    
    # Convert mask to float values and normalize so the matrix sums up to 1.0
    kernel = mask.astype(np.float32)
    kernel /= kernel.sum()
    
    return kernel


def convolve2d(image, kernel):
    height = len(image)
    width = len(image[0])

    kheight = len(kernel)
    kwidth = len(kernel[0])

    pad_h = kheight // 2
    pad_w = kwidth // 2

    result = []
    for i in range(height):
        result.append([0] * width)

    for i in range(height):
        for j in range(width):

            total = 0.0

            for ki in range(kheight):
                ii = i + ki - pad_h
                if ii < 0 or ii >= height:
                    continue

                for kj in range(kwidth):
                    jj = j + kj - pad_w
                    if jj < 0 or jj >= width:
                        continue

                    total += image[ii][jj] * kernel[ki][kj]

            result[i][j] = total

    return result

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

