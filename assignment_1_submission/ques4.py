import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from numba import njit
import os

out_dir = 'ques4_output_images'
os.makedirs(out_dir, exist_ok=True)

img = Image.open('colour_image.jpg')
img = np.asarray(img)
img_r = img[...,0]
img_g = img[...,1]
img_b = img[...,2]

rg = np.dstack((img_r, img_g, np.zeros_like(img_b)))
gb = np.dstack((np.zeros_like(img_r), img_g, img_b))
rb = np.dstack((img_r, np.zeros_like(img_g), img_b))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].imshow(img)
axes[0,0].set_title("input image")

axes[0,1].imshow(img_r, cmap="gray", vmin=0, vmax=255)
axes[0,1].set_title("red colours of image")

axes[1,0].imshow(img_g, cmap="gray", vmin=0, vmax=255)
axes[1,0].set_title("green colours ofimage")

axes[1,1].imshow(img_b, cmap="gray", vmin=0, vmax=255)
axes[1,1].set_title("blue colours ofimage")

plt.tight_layout()
plt.show()




def get_intensity(r,g,b):
    r = img[..., 0].astype(np.float32) / 255
    g = img[..., 1].astype(np.float32) / 255
    b = img[..., 2].astype(np.float32) / 255
    i = (r+g+b)/3
    return i 

def get_saturation(r,g,b):
    r = img[..., 0].astype(np.float32) / 255
    g = img[..., 1].astype(np.float32) / 255
    b = img[..., 2].astype(np.float32) / 255

    
    minimum = np.minimum(np.minimum(r, g), b)
    total = r + g + b

    # division by zero check
    saturation = np.zeros_like(total, dtype=np.float32)
    nonzero = total != 0

    saturation[nonzero] = (
        1 - (3 * minimum[nonzero] / total[nonzero])
    )

    return saturation

def get_hue(r,g,b):
    r = img[..., 0].astype(np.float32) / 255
    g = img[..., 1].astype(np.float32) / 255
    b = img[..., 2].astype(np.float32) / 255

    d1 = r-g
    d2 = r-b 
    d3 = g-b 
    sum = 0.5*(d1+d2)
    eps = 1e-8
    denominator = (d1**2) + d2*d3 +eps
    denominator = np.sqrt(denominator)
    ratio = sum/denominator
    ratio = np.clip(ratio,-1,1)

    theta_rad = np.arccos((ratio))
    theta = np.degrees(theta_rad) 

    theta = np.where(b > g , 360- theta, theta)

    return theta


def get_hsi_to_rgb(hue=180,s=0.5,intensity=0.5):

    if(np.isscalar(hue) == False):
        temp_arr = hue 
    elif(np.isscalar(s) == False):
        temp_arr = s
    else:
        temp_arr = intensity
    r = np.zeros_like(temp_arr, dtype=np.float64)
    g = np.zeros_like(temp_arr, dtype=np.float64)
    b = np.zeros_like(temp_arr, dtype=np.float64)


    hue = np.full_like(temp_arr, hue) if np.isscalar(hue) else hue
    s = np.full_like(temp_arr, s) if np.isscalar(s) else s
    intensity = np.full_like(temp_arr, intensity) if np.isscalar(intensity) else intensity

    for i in range(1200):
        for j in range(1600):
            currh = hue[i][j]
            currs = s[i][j]
            curri = intensity[i][j]
            if( 0 <= currh  and currh < 120):

                b[i][j] = curri*(1 - currs)
                r[i][j] = curri*(1 + (currs*np.cos(math.radians(currh)) / (np.cos(math.radians(60 - currh))) ) )
                g[i][j] = 3*curri - (r[i][j] + b[i][j])

            elif (120 <= currh and currh < 240 ): 
                r[i][j] = curri*(1 - currs)
                g[i][j] = curri*(1 + (currs*np.cos(math.radians(currh)) / (np.cos(math.radians(60 - currh))) ) )
                b[i][j] = 3*curri - (r[i][j] + g[i][j])
            else:
                g[i][j] = curri*(1 - currs)
                b[i][j] = curri*(1 + (currs*np.cos(math.radians(currh)) / (np.cos(math.radians(60 - currh))) ) )
                r[i][j] = 3*curri - (g[i][j] + b[i][j])

    # r = [normalise_to_255(r.max(),r.min(),i) for i in r ]
    # g = [normalise_to_255(g.max(),g.min(),i) for i in g ]
    # b = [normalise_to_255(b.max(),b.min(),i) for i in b ]

    # rgb = np.stack([r,g,b],axis =-1 )
    rgb = np.clip(np.stack([r, g, b], axis=-1) * 255, 0, 255).astype(np.uint8)
    return rgb

def normalise_to_255(maxx, minn ,val):
    val = (val - minn) / (maxx - minn )
    # print("returned val = ",val*255,'\t minn = ',minn,'\t maxx = ',maxx)
    return val * 255



def get_ycbcr_from_rgb(r,g,b):
    y = 0.299*r + 0.587*g +0.114*b 
    cb = -1*0.168*r - 0.331*g + 0.5*b + 128
    cr = 0.5*r -0.419*g -0.081*b +128
    return y,cb,cr 

def get_rgb_from_ycbcr(y=128,cb=128,cr=128):

    if(np.isscalar(y) == False):
        temp_arr = y
    elif(np.isscalar(cb) == False):
        temp_arr = cb
    else:
        temp_arr = cr
    
    y = np.full_like(temp_arr, y) if np.isscalar(y) else y
    cb = np.full_like(temp_arr, cb) if np.isscalar(cb) else cb
    cr = np.full_like(temp_arr, cr) if np.isscalar(cr) else cr


    r = y + 1.402*( cr - 128 )
    g = y - 0.3441*( cb - 128 ) - 0.7141* ( cr -128)
    b = y + 1.772*(cb - 128)
    rgb = np.stack([r,g,b],axis = -1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb 


@njit
def spatial_weight(di, dj, sigma_space):
    return np.exp(-(di**2 + dj**2) / (2 * sigma_space**2))

@njit
def color_weight(cb_center, cr_center, cb_neighbor, cr_neighbor, sigma_color):
    dist2 = (cb_center - cb_neighbor)**2 + (cr_center - cr_neighbor)**2
    return np.exp(-dist2 / (2 * sigma_color**2))

@njit
def bilateral_filter(y, cb, cr, w, sigma_space=10, sigma_colour=30):
    p = w // 2
    rows = len(y)
    cols = len(y[0])

    y_out_intensity_and_colour_combined = np.zeros_like(y, dtype=np.float64)
    y_out_intensity_only = np.zeros_like(y,dtype=np.float64)
    y_out_colour_only = np.zeros_like(y,dtype=np.float64)

    for i in range(rows):
        # print(i)
        for j in range(cols):

            cb_center = cb[i][j]
            cr_center = cr[i][j]

            weighted_sum = 0
            weight_total = 0

            intensity_sum = 0
            intensity_weight_total = 0 

            colour_sum = 0
            colour_weight_total = 0

            for di in range(-p, p+1):
                for dj in range(-p, p+1):
                    ni = i + di   
                    nj = j + dj   
                    if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                        continue

                    y_neighbor  = y[ni][nj]
                    cb_neighbor = cb[ni][nj]
                    cr_neighbor = cr[ni][nj]

                    w_space = spatial_weight(di, dj, sigma_space)
                    w_colour = color_weight(cb_center, cr_center, cb_neighbor, cr_neighbor, sigma_colour)

                    weight = w_space * w_colour

                    weighted_sum += weight * y_neighbor
                    weight_total += weight

                    intensity_sum += w_space * y_neighbor
                    intensity_weight_total += w_space  

                    colour_sum += w_colour * y_neighbor
                    colour_weight_total += w_colour

            y_out_intensity_and_colour_combined[i][j] = weighted_sum / weight_total
            y_out_intensity_only[i][j] = intensity_sum / intensity_weight_total
            y_out_colour_only[i][j] = colour_sum / colour_weight_total

    return y_out_intensity_and_colour_combined,y_out_intensity_only,y_out_colour_only


#hsi

intensity = get_intensity(img_r,img_g,img_b)
sat = get_saturation(img_r,img_g,img_b)
hue = get_hue(img_r,img_g,img_b)

# print("r = ",img_r,"    size = ",img_r.shape,"\n\n")
# print("i = ",intensity,"size = ",intensity.shape,"\n\n")
# print("s = ",sat,"size = ",sat.shape,"\n\n")
# print("h = ",hue,"size = ",hue.shape,"\n\n")

fig,axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].imshow(img)
axes[0,0].set_title("input image")

axes[0,1].imshow(sat,cmap='gray',vmin=0, vmax=1)
axes[0,1].set_title("saturation")

axes[1,0].imshow(intensity,cmap='gray')
axes[1,0].set_title('intensity')

axes[1,1].imshow(hue,cmap='gray',vmin=0,vmax=360)
axes[1,1].set_title('hue')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "HSI_plot.jpg"), dpi=150)
plt.show()







img_rgb_back_from_hue = get_hsi_to_rgb(hue)
img_rgb_back_from_hue1 = get_hsi_to_rgb(hue,1)
img_rgb_back_from_hue2 = get_hsi_to_rgb(hue,1,1)
img_rgb_back_from_hue3 = get_hsi_to_rgb(hue,0,0)


fig,axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].imshow(img_rgb_back_from_hue)
axes[0,0].set_title("rgb from hue , sat = 0.5, i = 0.5")

axes[0,1].imshow(img_rgb_back_from_hue1)
axes[0,1].set_title("rgb from hue , sat = 1, i = 0.5")

axes[1,0].imshow(img_rgb_back_from_hue2)
axes[1,0].set_title("rgb from hue , sat = 1, i = 1")

axes[1,1].imshow(img_rgb_back_from_hue3)
axes[1,1].set_title("rgb from hue , sat = 0, i = 0")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Image_back_from_Hue_alone.jpg"), dpi=150)
plt.show()


#from saturation

img_rgb_back_from_sat = get_hsi_to_rgb(s=sat)
img_rgb_back_from_sat1 = get_hsi_to_rgb(360,sat)
img_rgb_back_from_sat2 = get_hsi_to_rgb(360,sat,1)
img_rgb_back_from_sat3 = get_hsi_to_rgb(360,sat,0)


fig,axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].imshow(img_rgb_back_from_sat)
axes[0,0].set_title("rgb from sat , hue = 180, i = 0.5")

axes[0,1].imshow(img_rgb_back_from_sat1)
axes[0,1].set_title("rgb from sat , hue = 360, i = 0.5")

axes[1,0].imshow(img_rgb_back_from_sat2)
axes[1,0].set_title("rgb from sat , hue = 360, i = 1")

axes[1,1].imshow(img_rgb_back_from_sat3)
axes[1,1].set_title("rgb from sat , hue = 360, i = 0")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Image_back_from_Saturation_alone.jpg"), dpi=150)
plt.show()


#from intensity
img_rgb_back_from_intensity = get_hsi_to_rgb(intensity=intensity)
img_rgb_back_from_intensity1 = get_hsi_to_rgb(360,0.5,intensity)
img_rgb_back_from_intensity2 = get_hsi_to_rgb(360,1,intensity)
img_rgb_back_from_intensity3 = get_hsi_to_rgb(360,0,intensity)


fig,axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].imshow(img_rgb_back_from_intensity)
axes[0,0].set_title("rgb from intensity , hue = 180, s = 0.5")

axes[0,1].imshow(img_rgb_back_from_intensity1)
axes[0,1].set_title("rgb from intensity , hue = 360, s = 0.5")

axes[1,0].imshow(img_rgb_back_from_intensity2)
axes[1,0].set_title("rgb from intensity , hue = 360, s = 1")

axes[1,1].imshow(img_rgb_back_from_intensity3)
axes[1,1].set_title("rgb from intensity , hue = 360, s = 0")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Image_back_from_Intensity_alone.jpg"), dpi=150)
plt.show()



## ycbcr

y,cb,cr = get_ycbcr_from_rgb(img_r,img_g,img_b)

fig,axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].imshow(img)
axes[0,0].set_title('base image')

axes[0,1].imshow(y,cmap='gray')
axes[0,1].set_title('y image')

axes[1,0].imshow(cb,cmap='gray')
axes[1,0].set_title('cb image')

axes[1,1].imshow(cr,cmap='gray')
axes[1,1].set_title('cr image')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "YCbCR_channels_separated.jpg"), dpi=150)
plt.show()


rgb_ycbcr = get_rgb_from_ycbcr(y,cb,cr)
rgb_ycbcr1 = get_rgb_from_ycbcr(y)
rgb_ycbcr2 = get_rgb_from_ycbcr(cb=cb)
rgb_ycbcr3 = get_rgb_from_ycbcr(cr=cr)

fig,axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].imshow(rgb_ycbcr)
axes[0,0].set_title('base image rgb from ycbcr')

axes[0,1].imshow(rgb_ycbcr1)
axes[0,1].set_title('rgb from y only image')

axes[1,0].imshow(rgb_ycbcr2)
axes[1,0].set_title('rgb from cb only image')

axes[1,1].imshow(rgb_ycbcr3)
axes[1,1].set_title('rgb from cr only image')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Image_back_from_Y__Cb_and_Cr_alone_at_a_time.jpg"), dpi=150)
plt.show()



k_values = [2,5,10,20]
fig,axes = plt.subplots(2, 2, figsize=(12, 8))

for i in range(2):
    for j in range(2):
        sat_scaled = np.clip(sat*k_values[i*2+j],0,1)
        scaled_sat_rgb = get_hsi_to_rgb(hue,sat_scaled,intensity) 

        axes[i,j].imshow(scaled_sat_rgb)
        axes[i,j].set_title(f'image w saturation multiplied by k={k_values[i*2+j]}')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Images_for_saturation_multiplied_by_varying_factors.jpg"), dpi=150)
plt.show()



#part d

intensity_inverted = 1 - intensity
y_inverted = 255 - y

rgb_i_inverted = get_hsi_to_rgb(hue,sat,intensity_inverted)
ycbcr_y_inverted = get_rgb_from_ycbcr(y_inverted,cb,cr)

fig,axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(rgb_i_inverted)
axes[0].set_title('inverted intensity in hsi')

axes[1].imshow(ycbcr_y_inverted)
axes[1].set_title("inverted y in ycbcr")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Inverted_HSI_Instensity_and_Inverted_YCbCr_Y.jpg"), dpi=150)
plt.show()




#part e


y_change_colour_with_intensity , y_change_intensity, y_change_colour = bilateral_filter(y,cb,cr,21)

rgb_y_col_intensity = get_rgb_from_ycbcr(y_change_colour_with_intensity,cb,cr)
rgb_y_intensity = get_rgb_from_ycbcr(y_change_intensity,cb,cr)
rgb_y_colour = get_rgb_from_ycbcr(y_change_colour,cb,cr)


figs,axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].imshow(img)
axes[0,0].set_title('normal rgb image ')

axes[0,1].imshow(rgb_y_col_intensity)
axes[0,1].set_title('rgb image with filter weightage to both colour and intensity ')

axes[1,0].imshow(rgb_y_intensity)
axes[1,0].set_title('rgb image with filter only weighting intensity ')

axes[1,1].imshow(rgb_y_colour)
axes[1,1].set_title('rgb image with filter only weighing colour(cb,cr) ')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Brightness_Smoothing_Out_by_different_methods.jpg"), dpi=150)
plt.show()