import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import cmath
from q1functions import fft, fft2, ifft, padding, create_disc_kernel

# Example Usage: Generate a disc kernel with a radius of 2 pixels
disc_kernel = create_disc_kernel(radius=20)
padded_disc_kernel = padding(disc_kernel)


# 1. Get the dimensions of your padded matrix
height, width = padded_disc_kernel.shape
# 2. Create coordinate grids for x (columns) and y (rows)
y, x = np.indices((height, width))
# 3. Compute the (-1)^(x+y) sign matrix efficiently using the even/odd property
# (Since (-1)^n is 1 if n is even and -1 if n is odd)
sign_matrix = np.where((x + y) % 2 == 0, 1, -1)
# 4. Multiply element-wise to get the shifted matrix
shift_disc_kernel = padded_disc_kernel * sign_matrix



fft2_padded_disc_kernel = np.array(fft2(shift_disc_kernel))
abs_fft2_kernel = np.abs(fft2_padded_disc_kernel) 

img = Image.open('green_bottle.jpeg').convert('L')
img_matrix = np.array(img)
heigh, widt = img_matrix.shape
padded_img_matrix = padding(img_matrix)

# 1. Get the dimensions of your padded matrix
height, width = padded_img_matrix.shape
# 2. Create coordinate grids for x (columns) and y (rows)
y, x = np.indices((height, width))
# 3. Compute the (-1)^(x+y) sign matrix efficiently using the even/odd property
# (Since (-1)^n is 1 if n is even and -1 if n is odd)
sign1_matrix = np.where((x + y) % 2 == 0, 1, -1)
# 4. Multiply element-wise to get the shifted matrix
shift_padded_img_matrix = padded_img_matrix * sign1_matrix

fft2_padded_img_matrix = np.array(fft2(shift_padded_img_matrix))

log_spectrum = np.log(1 + np.abs(fft2_padded_img_matrix))

#Multiplication of Image and Filter in Frequency Domain
matrix_after_mul = fft2_padded_disc_kernel*fft2_padded_img_matrix
abs_mul = np.abs(matrix_after_mul)
log_abs_mul = np.log(1 + abs_mul)

#Taking IFFT
ifft_matrix_after_mul = ifft(matrix_after_mul)


# 1. Get the dimensions of your padded matrix
height, width = ifft_matrix_after_mul.shape
# 2. Create coordinate grids for x (columns) and y (rows)
y, x = np.indices((height, width))
# 3. Compute the (-1)^(x+y) sign matrix efficiently using the even/odd property
# (Since (-1)^n is 1 if n is even and -1 if n is odd)
sign2_matrix = np.where((x + y) % 2 == 0, 1, -1)
# 4. Multiply element-wise to get the shifted matrix
unshift_ifft = ifft_matrix_after_mul * sign2_matrix


real_ifft_matrix_after_mul = np.real(unshift_ifft)

unpadded_image = real_ifft_matrix_after_mul[0:heigh, 0:widt]

fig, axis = plt.subplots(2,3, figsize = (10,10))

axis[0,0].imshow(padded_img_matrix, cmap='gray')
axis[0,0].set_title(f"Padded_Image{padded_img_matrix.shape}")
axis[0,0].axis('on')  # Shows pixel coordinates on the borders
#axis[].show()

axis[0,1].imshow(log_spectrum, cmap='gray')
axis[0,1].set_title(f"FFT of Padded_Image{log_spectrum.shape}")
axis[0,1].axis('on')  # Shows pixel coordinates on the borders

axis[0,2].imshow(abs_fft2_kernel, cmap='gray')
axis[0,2].set_title(f"Filter Kernel{abs_fft2_kernel.shape}")
axis[0,2].axis('on')  # Shows pixel coordinates on the borders

axis[1,0].imshow(log_abs_mul, cmap='gray')
axis[1,0].set_title(f"Filtered in Frequency Domain")
axis[1,0].axis('on')  # Shows pixel coordinates on the borders

axis[1,1].imshow(real_ifft_matrix_after_mul, cmap='gray')
axis[1,1].set_title(f"Image After FFT, kernel, IFFT")
axis[1,1].axis('on')  # Shows pixel coordinates on the borders

axis[1,2].imshow(unpadded_image, cmap='gray')
axis[1,2].set_title(f"Unpadded_Image {unpadded_image.shape}")
axis[1,2].axis('on')  # Shows pixel coordinates on the borders


plt.savefig(f"q1b.png", dpi=130, bbox_inches="tight")
plt.show()
