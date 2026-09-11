import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import cmath
from q1functions import fft, fft2, ifft, padding
 
img = Image.open('green_bottle.jpeg').convert('L')
img_matrix = np.array(img)

print(img_matrix.shape)
print(img_matrix.dtype)

padded_matrix = padding(img_matrix)

fft_padded = fft2(padded_matrix)
ifft_fft_padded = ifft(fft_padded)
real_ifft_fft_padded = np.real(ifft_fft_padded)

fig, axis = plt.subplots(1,2,figsize = (8,6))

axis[0].imshow(padded_matrix, cmap='gray')
axis[0].set_title(f"Image Original{padded_matrix.shape}")
axis[0].axis('on')  # Shows pixel coordinates on the borders

axis[1].imshow(real_ifft_fft_padded, cmap='gray')
axis[1].set_title(f"Image After taking FFT then IFFT {real_ifft_fft_padded.shape}")
axis[1].axis('on')  # Shows pixel coordinates on the borders
plt.savefig(f"q1a.png", dpi=130, bbox_inches="tight")
plt.show()