import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numba import njit

@njit(cache=True)
def bilateral_filter(img_array, sigma_s, sigma_r, pad):
    rows, cols = img_array.shape
    window_size = 2 * pad + 1
    
    padded_rows = rows + 2 * pad
    padded_cols = cols + 2 * pad
    padded = np.zeros((padded_rows, padded_cols), dtype=np.float64)
    padded[pad:pad+rows, pad:pad+cols] = img_array
    
    for r in range(pad):
        padded[pad - 1 - r, pad:pad+cols] = img_array[r + 1, :]
        padded[padded_rows - pad + r, pad:pad+cols] = img_array[rows - 2 - r, :]
        
    for c in range(pad):
        padded[:, pad - 1 - c] = padded[:, pad + 1 + c]
        padded[:, padded_cols - pad + c] = padded[:, padded_cols - pad - 2 - c]
        
    spatial_weights = np.zeros((window_size, window_size), dtype=np.float64)
    for wr in range(window_size):
        for wc in range(window_size):
            di = wr - pad
            dj = wc - pad
            spatial_weights[wr, wc] = np.exp(-(di**2 + dj**2) / (2.0 * sigma_s**2))
            
    output = np.zeros((rows, cols), dtype=np.float64)
    two_sigma_r_sq = 2.0 * (sigma_r ** 2)
    
    for r in range(rows):
        for c in range(cols):
            center_val = img_array[r, c]
            weight_sum = 0.0
            filtered_val = 0.0
            
            for wr in range(window_size):
                for wc in range(window_size):
                    pixel_val = padded[r + wr, c + wc]
                    spatial_w = spatial_weights[wr, wc]
                    
                    intensity_diff = pixel_val - center_val
                    range_w = np.exp(-(intensity_diff**2) / two_sigma_r_sq)
                    
                    combined_w = spatial_w * range_w
                    weight_sum += combined_w
                    filtered_val += pixel_val * combined_w
                    
            if weight_sum > 0.0:
                output[r, c] = filtered_val / weight_sum
            else:
                output[r, c] = center_val
                
    return output

def base_detail_decomposition_bilateral(img_array, sigma_s=5.0, sigma_r=0.25, alpha=0.3, beta=1.4):
    eps = 1.0
    log_domain = np.log(img_array + eps)
    
    pad = int(np.ceil(3 * sigma_s))
    
    base_component = bilateral_filter(log_domain, sigma_s, sigma_r, pad)
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
    
    sigma_s_values = [1.0, 3.0, 7.0]
    sigma_r_values = [0.04, 0.12, 0.39]
    
    results = {}
    
    for s_val in sigma_s_values:
        for r_val in sigma_r_values:
            base_vis, detail_vis, final_out = base_detail_decomposition_bilateral(
                img_pixels, 
                sigma_s=s_val, 
                sigma_r=r_val, 
                alpha=0.3, 
                beta=1.4
            )
            results[(s_val, r_val)] = (base_vis, detail_vis, final_out)

    fig1, axes1 = plt.subplots(5, 4, figsize=(18, 22))
    fig1.suptitle('Bilateral Decomposition: Base and Detail Layers', fontsize=16, y=0.98)
    
    pair_count = 0
    for s_val in sigma_s_values:
        for r_val in sigma_r_values:
            base_vis, detail_vis, _ = results[(s_val, r_val)]
            
            row = pair_count // 2
            col_offset = (pair_count % 2) * 2
            
            axes1[row, col_offset].imshow(base_vis, cmap='gray')
            axes1[row, col_offset].set_title(f'Base (sigma_s={s_val}, sigma_r={r_val})')
            axes1[row, col_offset].axis('off')
            
            axes1[row, col_offset + 1].imshow(detail_vis, cmap='gray')
            axes1[row, col_offset + 1].set_title(f'Detail (sigma_s={s_val}, sigma_r={r_val})')
            axes1[row, col_offset + 1].axis('off')
            
            pair_count += 1

    axes1[4, 2].set_visible(False)
    axes1[4, 3].set_visible(False)

    plt.subplots_adjust(hspace=0.35, wspace=0.15, top=0.94, bottom=0.02, left=0.03, right=0.97)
    plt.savefig('bilateral_base_detail_comparison.jpg', dpi=150, bbox_inches='tight')

    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 15))
    fig2.suptitle('Bilateral Decomposition: Final Outputs Grid', fontsize=16, y=0.96)
    
    for s_idx, s_val in enumerate(sigma_s_values):
        for r_idx, r_val in enumerate(sigma_r_values):
            _, _, final_out = results[(s_idx, r_val)] if (s_idx, r_val) in results else results[(s_val, r_val)]
            
            ax = axes2[s_idx, r_idx]
            ax.imshow(final_out, cmap='gray')
            ax.set_title(f'sigma_s={s_val} | sigma_r={r_val}')
            ax.axis('off')

    plt.subplots_adjust(hspace=0.25, wspace=0.15, top=0.90, bottom=0.03, left=0.03, right=0.97)
    plt.savefig('bilateral_final_outputs_comparison.jpg', dpi=150, bbox_inches='tight')
    plt.show()