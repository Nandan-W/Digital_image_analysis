import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from q2functions import nearest_neighbor_resize, ideal_lowpass_filter, separable_butterworth_filter

# Load image
image = np.array(Image.open("newspaper.jpeg").convert("L"))

# Scale factor found experimentally
scale_factor = 26

# Butterworth orders
orders = [2, 5, 10, 20]

# Fourier transform
F = np.fft.fftshift(np.fft.fft2(image))

plt.figure(figsize=(14, 10))

for i, order in enumerate(orders):

    # Create Butterworth filter
    H = separable_butterworth_filter(image.shape,scale_factor,order)

    # Apply filter
    F_filtered = F * H

    # Inverse FFT
    filtered_image = np.fft.ifft2(np.fft.ifftshift(F_filtered))
    filtered_image = np.real(filtered_image)

    # Downsample
    resampled_image = nearest_neighbor_resize(filtered_image,scale_factor)

    # 1. Plot the Filtered Image on the LEFT (Column 1)
    plt.subplot(4, 2, 2 * i + 1)
    plt.imshow(filtered_image, cmap="gray", interpolation="nearest")
    plt.title(f"Filtered (n = {order})")
    plt.axis("off")

    # 2. Plot the Resampled Image on the RIGHT (Column 2)
    plt.subplot(4, 2, 2 * i + 2)
    plt.imshow(resampled_image, cmap="gray", interpolation="nearest")
    plt.title(f"Resampled (Scale = {scale_factor})")
    plt.axis("off")

plt.tight_layout()
plt.savefig(f"q2bbutter.png", dpi=130, bbox_inches="tight")
plt.show()