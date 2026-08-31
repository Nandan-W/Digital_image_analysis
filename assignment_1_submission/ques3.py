import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 

out_dir = 'ques3_output_images'
os.makedirs(out_dir, exist_ok=True)

def rgb_to_hsi_scratch(rgb_array):
    r = rgb_array[:, :, 0] / 255.0
    g = rgb_array[:, :, 1] / 255.0
    b = rgb_array[:, :, 2] / 255.0

    intensity = ((r + g + b) / 3.0) * 255.0

    min_rgb = np.minimum(np.minimum(r, g), b)
    sum_rgb = r + g + b

    saturation = np.zeros_like(intensity)
    nonzero_sum = sum_rgb > 0
    saturation[nonzero_sum] = (1.0 - (3.0 * min_rgb[nonzero_sum] / sum_rgb[nonzero_sum])) * 255.0

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b) * (g - b))

    valid_den = den > 1e-6
    h_temp = np.zeros_like(intensity)

    angle = np.arccos(np.clip(num[valid_den] / den[valid_den], -1.0, 1.0))
    angle_deg = np.degrees(angle)
    h_temp[valid_den] = angle_deg

    idx_b_greater = (b > g) & valid_den
    h_temp[idx_b_greater] = 360.0 - h_temp[idx_b_greater]

    hue = h_temp / 2.0

    return np.stack([hue, saturation, intensity], axis=-1)

def create_disk_kernel(radius):
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1.0
    kernel /= np.sum(kernel)
    return kernel

def convolve_from_scratch(image_array, kernel):
    img_h, img_w, num_channels = image_array.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    output_array = np.zeros_like(image_array, dtype=np.float32)

    for c in range(num_channels):
        channel = image_array[:, :, c]
        padded_channel = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

        for y in range(img_h):
            for x in range(img_w):
                roi = padded_channel[y:y+k_h, x:x+k_w]
                output_array[y, x, c] = np.sum(roi * kernel)

    return output_array


image_path = 'ques3_input_image.jpeg'  

pil_img = Image.open(image_path).convert('RGB')
img_np = np.array(pil_img)

# mask for green bottle

hsi_img = rgb_to_hsi_scratch(img_np)

# hue sat intensity thresholds for green bottle 
lower_green = np.array([35, 40, 30])
upper_green = np.array([85, 255, 255])

bottle_mask = (
    (hsi_img[:,:,0] >= lower_green[0]) & (hsi_img[:,:,0] <= upper_green[0]) &
    (hsi_img[:,:,1] >= lower_green[1]) & (hsi_img[:,:,1] <= upper_green[1]) &
    (hsi_img[:,:,2] >= lower_green[2]) & (hsi_img[:,:,2] <= upper_green[2])
)
bottle_mask_3d = np.expand_dims(bottle_mask, axis=-1)

# Configure the Blur Kernel Scale
bokeh_radius = 31
disk_kernel = create_disk_kernel(bokeh_radius)

# without gamma correction so standard sRGB space
blurred_no_gamma = convolve_from_scratch(img_np.astype(np.float32), disk_kernel)
blurred_no_gamma = np.clip(blurred_no_gamma, 0, 255).astype(np.uint8)
output_no_gamma = np.where(bottle_mask_3d, img_np, blurred_no_gamma)

# Linearizing image (Forward gamma transformation)
linear_img = (img_np / 255.0) ** 2.2

blurred_linear = convolve_from_scratch(linear_img, disk_kernel)

# 6. Return to display space (Inverse Gamma Transformation)
blurred_srgb = blurred_linear ** (1.0 / 2.2)
blurred_with_gamma = np.clip(blurred_srgb * 255.0, 0, 255).astype(np.uint8)
output_with_gamma = np.where(bottle_mask_3d, img_np, blurred_with_gamma)

plt.figure(figsize=(24, 5))

plt.subplot(1, 5, 1)
plt.title("1. Original Photo")
plt.imshow(img_np)
plt.axis('off')


plt.subplot(1, 5, 2)
plt.title("2. Bottle Mask (HSI)")
plt.imshow(bottle_mask, cmap='gray')
plt.axis('off')


plt.subplot(1, 5, 3)
plt.title("3. Linearized Image\n(Post-Gamma, Pre-Filter)")
plt.imshow(np.clip(linear_img, 0, 1))
plt.axis('off')


plt.subplot(1, 5, 4)
plt.title("4. Bokeh WITHOUT Gamma")
plt.imshow(output_no_gamma)
plt.axis('off')


plt.subplot(1, 5, 5)
plt.title("5. Bokeh WITH Gamma\n(Realistic)")
plt.imshow(output_with_gamma)
plt.axis('off')

plt.tight_layout()

plt.savefig(f"{out_dir}/{'output_image'}", dpi=150)
plt.show()

