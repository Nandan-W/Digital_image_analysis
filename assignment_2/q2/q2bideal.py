import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from q2functions import nearest_neighbor_resize, ideal_lowpass_filter

# Load image
image = np.array(Image.open("newspaper.jpeg").convert("L"))

# Choose downsampling factor
factor = 26
# Shifted Fourier transform
F = np.fft.fft2(image)
F_shifted = np.fft.fftshift(F)

# ideal rectangular LPF
H = ideal_lowpass_filter(image.shape,factor)

# Apply filter
F_filtered = F_shifted * H

# Inverse Fourier transform
filtered_image = np.fft.ifft2(np.fft.ifftshift(F_filtered))
filtered_image = np.real(filtered_image)

# Downsample
resampled_image = nearest_neighbor_resize(filtered_image,factor)

# Display
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(filtered_image, cmap="gray")
plt.title("After Ideal Low-Pass Filtering")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(resampled_image, cmap="gray")
plt.title("After Filtering + Downsampling(factor = 26)")
plt.axis("off")

plt.tight_layout()
plt.savefig(f"q2bideal.png", dpi=130, bbox_inches="tight")
plt.show()