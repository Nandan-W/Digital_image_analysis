import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from q1functions import fft2, ifft, padding2, create_disc_kernel, convolve2d, nearest_neighbor_resize

# Load, grayscale, downscale (nearest_neighbor_resize), pad to 256x256
# (padding2) -- all functions used here come from q1functions.py only.
img = Image.open("green_bottle.jpeg").convert("L")
img_matrix = np.array(img)
print("original grayscale shape:", img_matrix.shape)

scale_factor = 7
resized = nearest_neighbor_resize(img_matrix, scale_factor)
print("resized shape:", resized.shape)

padded_matrix = padding2(resized).astype(np.float64)
print("padded shape:", padded_matrix.shape)

radii = [5, 9, 15 ,19, 25]

# SPATIAL DOMAIN filtering -- convolve2d from q1functions.py
spatial_results = {}
spatial_times = {}

for r in radii:
    print(f"[spatial] radius={r} ...")
    t0 = time.perf_counter()
    kernel = create_disc_kernel(r)
    filtered = convolve2d(padded_matrix, kernel)
    dt = time.perf_counter() - t0
    spatial_results[r] = filtered
    spatial_times[r] = dt
    print(f"    time = {dt:.4f} s")

# FREQUENCY DOMAIN filtering, using only fft2/ifft/padding2 from q1functions.
# Image's own FFT doesn't depend on the kernel -> computed once, reused.
print("[frequency] computing image fft2 once (shared across all radii) ...")
t0 = time.perf_counter()
image_fft = np.array(fft2(padded_matrix))
t_image_fft = time.perf_counter() - t0
print(f"    image fft2 time = {t_image_fft:.4f} s  (one-time cost, not counted per-radius below)")

freq_results = {}
freq_times = {}

for r in radii:
    print(f"[frequency] radius={r} ...")
    t0 = time.perf_counter()
    kernel = create_disc_kernel(r)
    padded_kernel = padding2(kernel)
    kernel_fft = np.array(fft2(padded_kernel))

    product = kernel_fft * image_fft
    restored = ifft(product)
    restored_real = np.real(restored)
    dt = time.perf_counter() - t0

    freq_results[r] = restored_real
    freq_times[r] = dt
    print(f"    time = {dt:.4f} s")


freq_times = {r: t + t_image_fft for r, t in freq_times.items()}

# Single combined plot: 2 rows (spatial / frequency) x 3 cols (radii)
fig, axes = plt.subplots(2, 5, figsize=(15, 10))

for i, r in enumerate(radii):
    axes[0, i].imshow(spatial_results[r], cmap="gray")
    axes[0, i].set_title(f"Spatial domain, radius={r}\ntime = {spatial_times[r]:.3f} s")
    axes[0, i].axis("off")

    axes[1, i].imshow(freq_results[r], cmap="gray")
    axes[1, i].set_title(f"Frequency domain, radius={r}\ntime = {freq_times[r]:.3f} s")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("q1c.png", dpi=130, bbox_inches="tight")
plt.show()
plt.close()

spatial_values_list = list(spatial_times.values())      #dictionary to list
freq_values_list = list(freq_times.values())

plt.plot(radii, spatial_values_list, marker = 's', color = 'red')
plt.plot(radii, freq_values_list, marker = 'o', color = 'green')

plt.xlabel("Radius")
plt.ylabel("Time")
plt.title("Frequency vs Spatial Times")
plt.grid(True)
plt.savefig("q1cplot.png", dpi = 150, bbox_inches = "tight")
plt.show()

print("\nSummary:")
print(f"{'radius':>8} {'spatial (s)':>14} {'frequency (s)':>16}")
for r in radii:
    print(f"{r:>8} {spatial_times[r]:>14.4f} {freq_times[r]:>16.4f}")
print(f"\n(image fft2, shared, one-time): {t_image_fft:.4f} s")
