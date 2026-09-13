import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


img = Image.open('newspaper.jpeg').convert('L')
img_matrix = np.array(img)

fft = np.fft.fft2(img_matrix)
fft_shifted = np.fft.fftshift(fft)
mag_fft_shifted = np.abs(fft_shifted)
log_mag_fft_shifted = np.log(1 + mag_fft_shifted)

fig, axis = plt.subplots(1,2, figsize = (10,10))

axis[0].imshow(img_matrix, cmap='gray')
axis[0].set_title(f"Grayscale Image Original{img_matrix.shape}")
axis[0].axis('on')
#axis[].show()

axis[1].imshow(log_mag_fft_shifted, cmap='gray')
axis[1].set_title(f"Image Fourier Transform (log scaled){log_mag_fft_shifted.shape}")
axis[1].axis('on')  
plt.savefig(f"q2.png", dpi=130, bbox_inches="tight")
plt.show()