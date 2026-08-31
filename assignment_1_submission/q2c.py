import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numba import njit

@njit(cache=True)
def gaussian_blur(img_array, kernel, pad):
    rows, cols = img_array.shape
    kernel_size = kernel.shape[0]
    
    padded_rows = rows + 2 * pad
    padded_cols = cols + 2 * pad
    padded_img = np.zeros((padded_rows, padded_cols), dtype=np.float64)
    
    padded_img[pad:pad+rows, pad:pad+cols] = img_array
    
    for r in range(pad):
        padded_img[pad - 1 - r, pad:pad+cols] = img_array[r + 1, :]
        padded_img[padded_rows - pad + r, pad:pad+cols] = img_array[rows - 2 - r, :]
        
    for c in range(pad):
        padded_img[:, pad - 1 - c] = padded_img[:, pad + 1 + c]
        padded_img[:, padded_cols - pad + c] = padded_img[:, padded_cols - pad - 2 - c]

    blurred = np.zeros((rows, cols), dtype=np.float64)
    
    for r in range(rows):
        for c in range(cols):
            window_sum = 0.0
            for kr in range(kernel_size):
                for kc in range(kernel_size):
                    pixel_val = padded_img[r + kr, c + kc]
                    kernel_val = kernel[kr, kc]
                    window_sum += pixel_val * kernel_val
            blurred[r, c] = window_sum
            
    return blurred

def create_gaussian_kernel(sigma):
    radius = int(np.ceil(3 * sigma))
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)

def base_detail_decomposition_gaussian(img_array, sigma=4.0, alpha=0.3, beta=1.5):
    eps = 1.0
    log_domain = np.log(img_array + eps)
    
    kernel = create_gaussian_kernel(sigma)
    pad = kernel.shape[0] // 2
    
    base_component = gaussian_blur(log_domain, kernel, pad)
    detail_component = log_domain - base_component
    
    base_visual = np.clip(np.exp(base_component) - eps, 0, 255).astype(np.uint8)
    detail_visual = np.clip(128 + (detail_component * 128), 0, 255).astype(np.uint8)
    
    base_mean = np.mean(base_component)
    compressed_base = alpha * (base_component - base_mean) + base_mean
    enhanced_detail = beta * detail_component
    
    recomposed_log = compressed_base + enhanced_detail
    recomposed_linear = np.exp(recomposed_log) - eps
    
    final_output = np.clip(recomposed_linear, 0, 255).astype(np.uint8)
    
    return base_visual, detail_visual, final_output

if __name__ == "__main__":
    try:
        img = Image.open('image_night_n.jpeg').convert('L')
    except FileNotFoundError:
        exit()

    img_pixels = np.array(img, dtype=np.float64)
    sigma_values = [2.0, 4.0, 8.0, 16.0]
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    fig.suptitle('Base-Detail Decomposition across Sigmas', fontsize=16, y=0.97)
    
    for idx, current_sigma in enumerate(sigma_values):
        base_img, detail_img, final_img = base_detail_decomposition_gaussian(
            img_pixels, 
            sigma=current_sigma, 
            alpha=0.3, 
            beta=1.5
        )
        
        Image.fromarray(base_img).save(f'5_base_sigma_{current_sigma}.jpg')
        Image.fromarray(detail_img).save(f'5_detail_sigma_{current_sigma}.jpg')
        Image.fromarray(final_img).save(f'5_final_output_sigma_{current_sigma}.jpg')
        
        axes[idx, 0].imshow(base_img, cmap='gray')
        axes[idx, 0].set_title(f'Base (sigma={current_sigma})')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(detail_img, cmap='gray')
        axes[idx, 1].set_title(f'Detail (sigma={current_sigma})')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(final_img, cmap='gray')
        axes[idx, 2].set_title(f'Final (sigma={current_sigma})')
        axes[idx, 2].axis('off')
        
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.03, top=0.92, hspace=0.35, wspace=0.20)
    plt.savefig('comprehensive_decomposition_grid.jpg', dpi=150, bbox_inches='tight')
    plt.show()