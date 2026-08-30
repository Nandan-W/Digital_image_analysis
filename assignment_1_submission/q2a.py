import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_histogram_and_cdf(img_array):
    flat = img_array.flatten()
    
    hist = np.zeros(256, dtype=np.int32)
    for pixel in flat:
        hist[int(pixel)] += 1
        
    cdf = np.zeros(256, dtype=np.int32)
    running_sum = 0
    for i in range(256):
        running_sum += hist[i]
        cdf[i] = running_sum
        
    cdf_min = cdf[cdf > 0].min() if cdf.max() > 0 else 0
    cdf_max = cdf.max()
    
    cdf_normalized = np.zeros(256, dtype=np.uint8)
    if cdf_max - cdf_min > 0:
        for i in range(256):
            if cdf[i] > 0:
                cdf_normalized[i] = int(((cdf[i] - cdf_min) / (cdf_max - cdf_min)) * 255)
            else:
                cdf_normalized[i] = 0
                
    return cdf_normalized

def get_histogram_equalization(img_array):
    image = np.clip(img_array, 0, 255).astype(np.uint8)
    cdf_lookup = get_histogram_and_cdf(image)
    equalized_image = np.zeros_like(image)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            old_intensity = image[row, col]
            equalized_image[row, col] = cdf_lookup[old_intensity]

    return equalized_image

def plot_all_histograms(original, standard, log_method, gamma_list, gamma_outputs):
    num_gammas = len(gamma_list)
    total_rows = max(3, num_gammas)
    
    fig, axes = plt.subplots(total_rows, 2, figsize=(15, 4.5 * total_rows))
    fig.suptitle('Histogram Comparison', fontsize=16, y=0.98)
    
    axes[0, 0].hist(original.flatten(), bins=256, range=(0, 256), color='#4A4A4A', alpha=0.85)
    axes[0, 0].set_title('Original', fontsize=12)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
    
    axes[1, 0].hist(standard.flatten(), bins=256, range=(0, 256), color='#1f77b4', alpha=0.85)
    axes[1, 0].set_title('Standard HE', fontsize=12)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.5)
    
    axes[2, 0].hist(log_method.flatten(), bins=256, range=(0, 256), color='#2ca02c', alpha=0.85)
    axes[2, 0].set_title('Log + HE', fontsize=12)
    axes[2, 0].grid(axis='y', linestyle='--', alpha=0.5)
    
    for r in range(3, total_rows):
        axes[r, 0].set_visible(False)
        
    for idx, gamma_val in enumerate(gamma_list):
        gamma_data = gamma_outputs[idx]
        axes[idx, 1].hist(gamma_data.flatten(), bins=256, range=(0, 256), color='#d62728', alpha=0.85)
        axes[idx, 1].set_title(f'Gamma (gamma={gamma_val}) + HE', fontsize=12)
        axes[idx, 1].grid(axis='y', linestyle='--', alpha=0.5)

    for row in range(total_rows):
        for col in range(2):
            ax = axes[row, col]
            if ax.get_visible():
                ax.set_xlabel('Pixel Intensity')
                ax.set_ylabel('Pixel Count')

    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.05, top=0.93, hspace=0.55, wspace=0.25)
    plt.savefig('comprehensive_histogram_analysis.jpg', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    try:
        img = Image.open('image_night_n.jpeg').convert('L')
    except FileNotFoundError:
        exit()
    
    pixels = np.array(img, dtype=np.float64)

    standard_he = get_histogram_equalization(pixels)
    Image.fromarray(standard_he).save('1_standard_he.jpg')

    r_max = np.max(pixels)
    c_log = 255.0 / np.log(1.0 + r_max) if r_max > 0 else 1.0
    log_pixels = c_log * np.log(1.0 + pixels)
    log_he = get_histogram_equalization(log_pixels)
    Image.fromarray(log_he).save('2_log_pre_he.jpg')

    gamma_list = [0.5, 0.8, 1.2, 1.5, 2.2]
    gamma_output_list = []
    
    for gamma in gamma_list:
        gamma_pixels = 255.0 * np.power(pixels / 255.0, gamma)
        gamma_he = get_histogram_equalization(gamma_pixels)
        Image.fromarray(gamma_he).save(f'3_gamma_pre_he_gamma_as_{gamma}.jpg')
        gamma_output_list.append(gamma_he)

    plot_all_histograms(pixels, standard_he, log_he, gamma_list, gamma_output_list)