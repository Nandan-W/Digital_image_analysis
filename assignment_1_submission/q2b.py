import numpy as np
from PIL import Image
from numba import njit

@njit(cache=True)
def clahe_core(padded_img, rows, cols, pad, window_size, clip_threshold):
    num_pixels = window_size * window_size
    output = np.zeros((rows, cols), dtype=np.uint8)
    
    hist = np.zeros(256, dtype=np.int32)
    clipped_hist = np.zeros(256, dtype=np.int32)
    cdf = np.zeros(256, dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            hist[:] = 0
            
            for wr in range(window_size):
                for wc in range(window_size):
                    pixel_val = padded_img[r + wr, c + wc]
                    hist[pixel_val] += 1
            
            excess = 0
            for i in range(256):
                if hist[i] > clip_threshold:
                    excess += (hist[i] - clip_threshold)
                    clipped_hist[i] = clip_threshold
                else:
                    clipped_hist[i] = hist[i]
            
            redistribution_amount = excess // 256
            remainder = excess % 256
            
            for i in range(256):
                clipped_hist[i] += redistribution_amount
                if i < remainder:
                    clipped_hist[i] += 1
            
            running_sum = 0
            cdf_min = -1
            for i in range(256):
                running_sum += clipped_hist[i]
                cdf[i] = running_sum
                if cdf_min == -1 and running_sum > 0:
                    cdf_min = running_sum
            
            center_val = padded_img[r + pad, c + pad]
            cdf_max = cdf[255]
            
            if cdf_max == cdf_min:
                output[r, c] = center_val
            else:
                val = (cdf[center_val] - cdf_min) / (num_pixels - cdf_min) * 255.0
                output[r, c] = np.uint8(val + 0.5)
                
    return output

def optimized_clahe_for_streetlight(img_array, window_size=121, clip_limit=2.0):
    img = np.clip(img_array, 0, 255).astype(np.uint8)
    rows, cols = img.shape
    pad = window_size // 2
    
    padded_img = np.pad(img, pad, mode='reflect')
    
    num_pixels = window_size * window_size
    ideal_bin_height = num_pixels / 256.0
    clip_threshold = max(1, int(clip_limit * ideal_bin_height))
    
    return clahe_core(padded_img, rows, cols, pad, window_size, clip_threshold)

if __name__ == "__main__":
    try:
        img = Image.open('image_night_n.jpeg').convert('L')
    except FileNotFoundError:
        exit()

    img_pixels = np.array(img, dtype=np.float64)
    
    window_size_list = [15, 31, 45, 61, 91, 121] 
    for window_size in window_size_list:
        clahe_result = optimized_clahe_for_streetlight(img_pixels, window_size, clip_limit=2.0)
        output_filename = f'best_local_clahe_output_window_size_{window_size}_.jpg'
        Image.fromarray(clahe_result).save(output_filename)