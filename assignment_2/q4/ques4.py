import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt
import math
import os


os.makedirs('output_images_ques4', exist_ok=True)

img_g = Image.open('book.jpg').convert('L')

img_g = np.asarray(img_g)
h,w = img_g.shape

fig =plt.figure()
plt.imshow(img_g,cmap='gray')
plt.title("pick coordinate 'a' (for initial position of translation point)")
p = plt.ginput(1,-1,True)
plt.close(fig)

fig = plt.figure()
plt.imshow(img_g,cmap='gray')
plt.title("pick coordinate 'b' (for final position of translation point)")
q = plt.ginput(1,-1,True)
plt.close(fig)



# now tan(b/a) = slope , so b = tan-1(slope)*a = tana(slope)*a = a*k , k is any value obtained
# and dist between them is d = (a^2 + b^2)^0.5 = ( ( a^2(1+k^2) )^0.5 )
# so a = d / ((1 + k ^2 )^0.5) 
# and so b = k*a
def get_vecs_a_and_b(coords1, coords2):
    ap, bp = coords1[0]
    aq, bq = coords2[0]
    a = aq - ap
    b = bq - bp
    return [a, b],

blur_vec_manual = get_vecs_a_and_b(p,q)[0]
a,b = blur_vec_manual
print("blur vector a and b calculated from manual point picking is = ", blur_vec_manual)
print("estimate blur length = ",(a**2 + b**2)**0.5)


G_img = np.fft.fft2(img_g)
img_log_spectrum = np.log(1 + np.abs(np.fft.fftshift(G_img)))

u = np.fft.fftfreq(w)
v = np.fft.fftfreq(h)

U,V = np.meshgrid(u,v)
q = a*U + b*V

# H = integral from 0 to 1 (range of t for a and b) h(x,y)*e^(-1*j*pi*(ua+bv)) * dx
# which gives H = sinc((au+bv)*pi)/(pi*(au+bv))* e^(-j*pi*(au+bv))
H = np.exp(-1j*np.pi*q)*np.sinc(q)

# |H(u,v)|
H_abs = np.fft.fftshift(np.abs(H))
img_log_spectrum = np.log(1+np.fft.fftshift(np.abs(G_img)))

fig,axes = plt.subplots(1,2,figsize=(10,10))
axes[0].imshow(img_log_spectrum,cmap='gray')
axes[0].set_title("shifted log spectrum of img FFT")

axes[1].imshow(H_abs,cmap='gray')
axes[1].set_title("visualization of spectrum |H(u,v)|")
plt.savefig('output_images_ques4/log_spectrum_of_image_FFT_and_abs_H_uv.png',dpi=150,bbox_inches='tight')
plt.show()


# # part b , inverse transform func for motion blur
eps = 1e-0
F_hat = G_img/ (eps + H)

inverse_log_spectrum = np.log(1 + np.abs(np.fft.fftshift(F_hat)))
F_hat_abs = np.fft.fftshift(np.abs(F_hat))



plt.imshow(inverse_log_spectrum,cmap='gray')
plt.title("shifted log spectrum of inverse tranform FFT")
plt.savefig('output_images_ques4/log_spectrum_of_Inverse_Transform_FFT.png',dpi=150,bbox_inches='tight')
plt.show()


eps_ranges = [1e-0, 1e-1, 1e-2, 1e-3]
inverse_filter_output_images = []
for eps in eps_ranges:
    F_hat = G_img / (H + eps)
    f_hat = np.real(np.fft.ifft2(F_hat))
    inverse_filter_output_images.append(f_hat)

m = len(inverse_filter_output_images)
fig,axes = plt.subplots(m//2, 2, figsize=(10,10))
for i in range(m//2):
    vmin, vmax = np.percentile(inverse_filter_output_images[2*i], [1,99])
    axes[i,0].imshow(inverse_filter_output_images[2*i], cmap='gray', vmin=vmin, vmax=vmax)
    axes[i,0].set_title(f'inverse filter applied, eps={eps_ranges[2*i]}')
    
    axes[i,1].imshow(inverse_filter_output_images[2*i+1], cmap='gray', vmin=vmin, vmax=vmax)
    axes[i,1].set_title(f'inverse filter applied, eps={eps_ranges[2*i+1]}')
plt.savefig('output_images_ques4/image_deblurred_by_inverse_filters.png',dpi=150,bbox_inches='tight')
plt.show()


## part c wiener deblur
# W = H*(u,v) / (|H(u,v)|^2 + K)
def wiener_deblurr(H,K):
    H_conj = np.conj(H)
    denom = np.square(np.abs(H)) + K 
    return H_conj / denom

# k = Sn / Sf ~ 1/psnr
K_vals = [1e0, 1e-1, 1e-2, 1e-3, 1e1, 1e2]
wiener_outputs = []
for k_val  in K_vals:
    W = wiener_deblurr(H,k_val)
    F_hat_wiener = W * G_img
    f_hat_wiener = np.real(np.fft.ifft2(F_hat_wiener))

    wiener_outputs.append(f_hat_wiener)

n = len(wiener_outputs)

fig,axes = plt.subplots(n//3, 3, figsize = (10,10))
for i in range(n//3):
    axes[i,0].imshow(wiener_outputs[3*i],cmap='gray')
    axes[i,0].set_title(f"wiener deblur for K = {K_vals[3*i]}")

    axes[i,1].imshow(wiener_outputs[3*i + 1],cmap='gray')
    axes[i,1].set_title(f"wiener deblur for K = {K_vals[3*i + 1]}")

    axes[i,2].imshow(wiener_outputs[3*i + 2],cmap='gray')
    axes[i,2].set_title(f"wiener deblur for K = {K_vals[3*i + 2]}")
plt.savefig('output_images_ques4/Wiener_filter_deblurred_output_images.png',dpi=150,bbox_inches='tight')
plt.show()


## part d
# wiener but with lamda weighted K

def get_dx_dy_freq(h,w):
    dx = np.zeros((h,w))
    dx[0,0] = -1
    dx[0,1] = 1
    Dx = np.fft.fft2(dx)

    dy = np.zeros((h,w))
    dy[0,0] = -1
    dy[1,0] = 1
    Dy = np.fft.fft2(dy)
    return Dx,Dy

def regularized_deconv_wiener(H,lamda,reg_term):
    H_conj = np.conj(H)
    denom = np.abs(H)**2 + lamda*reg_term
    return H_conj / denom

Dx, Dy = get_dx_dy_freq(h,w)
reg_term = np.abs(Dx)**2 + np.abs(Dy)**2

lamda_array = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
regularized_deconv_wiener_outputs = []
    
for lamda in lamda_array:
    W_regularised = regularized_deconv_wiener(H,lamda,reg_term)
    F_hat = W_regularised * G_img
    f_hat = np.real(np.fft.ifft2(F_hat))

    regularized_deconv_wiener_outputs.append(f_hat)

n = len(regularized_deconv_wiener_outputs)

fig, axes = plt.subplots(n//2,2,figsize=(10,10))

for i in range(n//2):
    axes[i,0].imshow(regularized_deconv_wiener_outputs[2*i],cmap='gray')
    axes[i,0].set_title(f"regularized deconvolution for lamda = {lamda_array[2*i]}")

    axes[i,1].imshow(regularized_deconv_wiener_outputs[2*i+1],cmap='gray')
    axes[i,1].set_title(f"regularized deconvolution for lamda = {lamda_array[2*i+1]}")
plt.savefig('output_images_ques4/regularized_deconvolution_deblurred_images.png',dpi=150,bbox_inches='tight')
plt.show()