import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import math
import os


os.makedirs('output_images_ques3', exist_ok=True)


input_img_address = 'ques_3_kodak_input_image.png'
img = Image.open(input_img_address).convert('L')
img = np.asarray(img)
h,w= img.shape

psnr = 20
# psnr is 10 * log (base 10 )(max range value / variance i.e. sigma sq )

def get_variance_from_psnr(psnr,max_val):
    return (max_val**2) / (10**(psnr/10))

max_val = 255
sigma_sq = get_variance_from_psnr(psnr,255)
sigma = sigma_sq ** (0.5)

zer_error_handler = 1e-9

def add_gaussian_noise(img_arr, sigma, mean=0):
    shape = img_arr.shape
    total = np.zeros(shape)
    
    # each is uniform in [0,1), mean 0.5, var 1/12
    for i in range(12):
        total += np.random.rand(*shape)   
    
    z = total - 6   
    # sum of 12 uniforms has mean 6, so subtracting to center at 0
    # variance of sum = 12 * (1/12) = 1, so z is N(0,1)

    noise = mean + sigma * z
    return noise

img_noise = add_gaussian_noise(img,sigma)
a = img_noise-img
b = -1*img_noise + img

test = add_gaussian_noise(np.zeros(200000), sigma=1)
print(test.mean(), test.var())   
plt.hist(test, bins=100, density=True)
plt.savefig('output_images_ques3/Histogram_of_Gaussian_noise_added.png',dpi=150,bbox_inches='tight')
plt.show()

fig,axes = plt.subplots(2,2, figsize = (10,10))
axes[0,0].imshow(img,cmap='gray')
axes[0,0].set_title('input image')

axes[0,1].imshow(img_noise,cmap='gray')
axes[0,1].set_title('noise image')

axes[1,0].imshow(img+img_noise,cmap='gray')
axes[1,0].set_title('noise added to image')

axes[1,1].imshow(a,cmap='gray')
axes[1,1].set_title('noise minus image')
plt.savefig('output_images_ques3/Input_Image_and_noise_and_their_combination_sum_and_difference.png',dpi=150,bbox_inches='tight')
plt.show()

img_g = img + img_noise

def psnr_of_image(noisy_img, img_f,max_val = 255):
    mse = np.mean((noisy_img - img_f)**2)
    return 10*np.log10(max_val*max_val/(mse ))

def estimate_var_patch(g, y0, y1, x0, x1):
    patch = g[y0:y1, x0:x1]
    return np.var(patch)

print(psnr_of_image(img_g,img))

est_var_by_patch = 0 
n_count = int(input('enter no of patches to check -->  '))
i = 0
while( n_count > i ):
    fig = plt.figure()
    plt.imshow(img_g,cmap='gray')
    plt.title(f"Click 2 corners of a flat patch ({i+1}/{n_count})")
    input_coords = plt.ginput(2,-1,True)
    plt.close(fig)

    if len(input_coords) < 2:
        print("Didn't get 2 clicks, skipping this round")
        continue

    a1 = int(input_coords[0][0])
    b1 = int(input_coords[0][1])
    a2 = int(input_coords[1][0])
    b2 = int(input_coords[1][1])
    x1 = min(a2,a1)
    x2 = max(a2,a1)
    y1 = min(b2,b1)
    y2 = max(b2,b1)
    
    a= estimate_var_patch(img_g, y1, y2, x1, x2)
    est_var_by_patch += a 
    print("variance of this patch  = ",a)
    i += 1
print("estimated var by patch method:", est_var_by_patch/n_count, " true:", sigma_sq)


def estimate_var_laplacian(g):
    L = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float64)
    H, W = g.shape
    lap = np.zeros((H-2, W-2))
    for i in range(H-2):
        for j in range(W-2):
            lap[i,j] = np.sum(g[i:i+3, j:j+3] * L)
    return np.sum(lap**2) / (36 * (H-2) * (W-2))

est_var_by_lap = estimate_var_laplacian(img_g)
print("estimated var by laplacian:", est_var_by_lap, "true:", sigma_sq)


# part b 

def apply_mean_filter(img,mean_filter_dim):
    filter = np.ones((mean_filter_dim,mean_filter_dim))
    filter /= (mean_filter_dim * mean_filter_dim)

    h,w = img.shape
    n = mean_filter_dim
    padding_size = n//2

    padded_img = np.pad(img, pad_width = padding_size,mode='constant',constant_values = 0)
    ans_img = np.zeros_like(img)
    for i in range(padding_size,h+padding_size):
        for j in range(padding_size,w+padding_size):
            ans_img[i-padding_size][j-padding_size] =  (padded_img[i-padding_size:i+padding_size + 1 ,j-padding_size:j+padding_size +1 ] * filter).sum()
    return ans_img

def apply_median_filter(img,filter_size):
    h,w = img.shape
    n = filter_size
    padding_size = n//2

    padded_img = np.pad(img,pad_width=padding_size,mode='constant',constant_values=0)
    ans_img = np.zeros_like(img)

    for i in range(padding_size,h+padding_size):
        for j in range(padding_size,w+padding_size):
            curr_window = padded_img[i-padding_size:i+padding_size+1,j-padding_size:j+padding_size+1]
            curr_window = curr_window.flatten()
            curr_window.sort()
            ans_img[i-padding_size][j-padding_size] = curr_window[(n*n) // 2]    
    return ans_img

filter_lengths = [1, 3 , 5, 7, 9, 11,15,19]
psnr_means = []
psnr_medians = []
mean_filtered_imgs = []
median_filtered_imgs = []

for i in filter_lengths:
    mean_filter_dim = i
    median_filter_dim = i

    mean_filtered_img = apply_mean_filter(img_g, mean_filter_dim)
    median_filtered_img = apply_median_filter(img_g, median_filter_dim)

    mean_filtered_imgs.append(mean_filtered_img)
    median_filtered_imgs.append(median_filtered_img)

    psnr_mean = psnr_of_image(mean_filtered_img,img)
    psnr_median = psnr_of_image(median_filtered_img,img)

    psnr_means.append(psnr_mean)
    psnr_medians.append(psnr_median)

filter_count = len(filter_lengths)

fig,axes = plt.subplots(filter_count // 2,4,figsize=(20,20))
for i in range(filter_count):
    axes[i//2,(2*i) % 4 ].imshow(mean_filtered_imgs[i],cmap='gray')
    axes[i//2,(2*i) % 4 ].set_title(f'mean filtered by {filter_lengths[i]} x {filter_lengths[i]} filter ')
    axes[i//2,(2*i) % 4].axis('off')

    axes[i//2,(2*i+1 )% 4 ].imshow(median_filtered_imgs[i],cmap='gray')
    axes[i//2,(2*i+1) % 4].set_title(f'median filtered by {filter_lengths[i]} x {filter_lengths[i]} filter ')
    axes[i//2,(2*i+1) % 4].axis('off')
plt.savefig('output_images_ques3/Results_of_mean_and_median_filter_denoising_for_different_filter_sizes.png',dpi=150,bbox_inches='tight')
plt.show()


best_m_idx_mean = np.argmax(psnr_means)
best_m_idx_median = np.argmax(psnr_medians)
print(f"best results for peak signal to noise ratio for mean filter comes for filter windows size = {filter_lengths[best_m_idx_mean]} x {filter_lengths[best_m_idx_mean]} ")
print(f"best results for peak signal to noise ratio for median filter comes for filter windows size = {filter_lengths[best_m_idx_median]} x {filter_lengths[best_m_idx_median]} ")

fig, axes = plt.subplots(1,2, figsize=(10,5))
axes[0].imshow(mean_filtered_imgs[best_m_idx_mean], cmap='gray')
axes[0].set_title(f'Best mean filter, m={filter_lengths[best_m_idx_mean]}')
axes[1].imshow(median_filtered_imgs[best_m_idx_median], cmap='gray')
axes[1].set_title(f'Best median filter, m={filter_lengths[best_m_idx_median]}')
plt.savefig('output_images_ques3/Best_mean_and_median_filters_found.png',dpi=150,bbox_inches='tight')
plt.show()


#now psnr of mean vs median for different m x m filters
plt.plot(filter_lengths,psnr_means,marker='o',color='green', label='psnr variance for images filtered by mean filter vs filter size m (m x m)')
plt.plot(filter_lengths,psnr_medians,marker='s',color='red', label='psnr variance for images filtered by median filter vs filter size m (m x m)')
plt.xlabel('filter size')
plt.ylabel('PSNR')
plt.title('mean and median filter psnr for varying size of m ')
plt.grid(True)
plt.legend()
plt.savefig('output_images_ques3/PSNR_varying_for_Mean_vs_Median.png',dpi=150,bbox_inches='tight')
plt.show()



# part c
#wiener filtering

G = np.fft.fft2(img_g)

# constant K wiener filter
# F_hat = G / (1+K)

def wiener_constant(G,K):
    return G / (1+K)

K0 = (h*w*sigma_sq) / np.sum(img_g.astype(np.float64)**2)
print("K0 initial guess --> ", K0)

k_values = [0.01*K0, 0.1*K0, 0.5*K0, K0, 2*K0, 10*K0, 50*K0]
psnr_const_k = []
restored_const_k = []

for K in k_values:
    F_hat = wiener_constant(G,K)
    f_hat = np.real(np.fft.ifft2(F_hat))
    p = psnr_of_image(f_hat,img)
    psnr_const_k.append(p)
    restored_const_k.append(f_hat)
    print("K = ", K, " psnr = ", p)

best_k_idx = np.argmax(psnr_const_k)
print("best K --> ", k_values[best_k_idx], " psnr --> ", psnr_const_k[best_k_idx])

fig,axes = plt.subplots(1,len(k_values),figsize=(20,5))
for i in range(len(k_values)):
    axes[i].imshow(restored_const_k[i],cmap='gray')
    axes[i].set_title(f'K = {k_values[i]:.2f} \n psnr = {psnr_const_k[i]:.2f}')
plt.savefig('output_images_ques3/Wiener_filter_denoised_images_for_different_values_of_K_Noise_to_signal_ratio.png',dpi=150,bbox_inches='tight')
plt.show()


# frequency dependent wiener filter
# S_eta[u,v] is flat since noise is white so S_eta = h*w*sigma_sq  
# S_f[u,v] approx max(|G|^2 - S_eta , 0)
# W = S_f / (S_f + S_eta)

def wiener_adaptive(G,sigma_sq,h,w):
    S_eta = h*w*sigma_sq
    S_f = np.maximum(np.abs(G)**2 - S_eta, 0)
    eps = 1e-8
    W = S_f / (S_f + S_eta + eps)
    return W*G

F_hat_adaptive = wiener_adaptive(G,sigma_sq,h,w)
f_hat_adaptive = np.real(np.fft.ifft2(F_hat_adaptive))
psnr_adaptive = psnr_of_image(f_hat_adaptive,img)
print("adaptive wiener psnr --> ", psnr_adaptive)

fig,axes = plt.subplots(1,3,figsize=(15,5))
axes[0].imshow(img,cmap='gray')
axes[0].set_title('original image')

axes[1].imshow(restored_const_k[best_k_idx],cmap='gray')
axes[1].set_title(f'best constant K wiener \n K = {k_values[best_k_idx]:.2f}, psnr = {psnr_const_k[best_k_idx]:.2f}')

axes[2].imshow(f_hat_adaptive,cmap='gray')
axes[2].set_title(f'adaptive wiener \n psnr = {psnr_adaptive:.2f}')
plt.savefig('output_images_ques3/Best_Wiener_filter_and_adaptive_wiener_filter.png',dpi=150,bbox_inches='tight')
plt.show()