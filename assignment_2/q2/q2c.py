import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from q2functions import create_notch_filter
# Load image
image = np.array(Image.open("newspaper.jpeg").convert("L"))
print(image.shape)

# Fourier transform
F = np.fft.fft2(image)
F_shifted = np.fft.fftshift(F)

# Show original spectrum
spectrum = np.log(1 + np.abs(F_shifted))

# Manually selected unwanted frequencies
#notch_points = [(803,796), (766,773), (735,742), (828,775), (689,804), (726,827), (757,858), (664,825)]

#notch_points = [(832, 718), (766, 774),(769, 716), (829, 776),(802, 688), (796, 804),(863, 749), (735, 743),]

notch_points = [
    # inner ring (fundamental frequencies)
    (832, 718), (766, 774),
    (769, 716), (829, 776),
    (802, 688), (796, 804),
    (863, 749), (735, 743),

    # outer ring (2nd-order harmonics of the same lattice)
    (740, 686), (858, 806),
    (735, 804), (863, 688),
    (827, 835), (771, 657),
    (764, 833), (834, 659),
    (704, 713), (894, 779),
    (702, 773), (896, 719),
]

# Adjust this manually
radius = 8

# Construct filter
H = create_notch_filter(image.shape,notch_points,radius)

# Apply filter
F_filtered = F_shifted * H

# Visualize filtered spectrum WITH log
filtered_spectrum = np.log(1 + np.abs(F_filtered))

# Inverse Fourier transform
filtered_image = np.fft.ifft2(np.fft.ifftshift(F_filtered))
filtered_image = np.real(filtered_image)

fig, axis = plt.subplots(2,2, figsize = (20,20))

# Show original spectrum
axis[0,0].imshow(spectrum, cmap="gray")
axis[0,0].set_title("Original Fourier Spectrum")
axis[0,0].axis("off")

# Visualize filter WITHOUT log
axis[0,1].imshow(H, cmap="gray")
axis[0,1].set_title("Notch-Reject Filter")
axis[0,1].axis("off")

axis[1,0].imshow(filtered_spectrum, cmap="gray")
axis[1,0].set_title("Spectrum After Notch Filtering")
axis[1,0].axis("off")

# Show final image
axis[1,1].imshow(filtered_image, cmap="gray")
axis[1,1].set_title("Final Image After Notch Filtering")
axis[1,1].axis("off")
plt.savefig(f"q2c.png", dpi=130, bbox_inches="tight")
plt.show()