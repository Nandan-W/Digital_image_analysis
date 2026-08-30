import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import os
from numba import njit

out_dir = 'ques1_output_images'
os.makedirs(out_dir, exist_ok=True)

left_img_path = r'imgL.jpg'
center_img_path = r'imgM.jpg'
right_img_path = r'imgR.jpg'

imgM = Image.open(center_img_path)
imgL = Image.open(left_img_path)
imgR = Image.open(right_img_path)

imgM_wt, imgM_ht = imgM.size
imgL_wt, imgL_ht = imgL.size
imgR_wt, imgR_ht = imgR.size

imgM_color = np.asarray(imgM)
imgL_color = np.asarray(imgL)
imgR_color = np.asarray(imgR)

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(imgL_color)
axes[0].set_title("left input")
axes[1].imshow(imgM_color)
axes[1].set_title("middle input")
axes[2].imshow(imgR_color)
axes[2].set_title("right input")
plt.tight_layout()
plt.savefig(f"{out_dir}/01_input_images_colour.png", dpi=150)
plt.show()


def to_grayscale(img):
    return (0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]).astype(np.uint8)


imgM_gray = to_grayscale(imgM_color)
imgL_gray = to_grayscale(imgL_color)
imgR_gray = to_grayscale(imgR_color)

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(imgL_gray, cmap='gray')
axes[0].set_title("left grayscale")
axes[1].imshow(imgM_gray, cmap='gray')
axes[1].set_title("middle grayscale")
axes[2].imshow(imgR_gray, cmap='gray')
axes[2].set_title("right grayscale")
plt.tight_layout()
plt.savefig(f"{out_dir}/02_input_images_grayscale.png", dpi=150)
plt.show()

#@njit
def ncc_match(patch, image):
    ph, pw = patch.shape
    ih, iw = image.shape
    out_h = ih - ph + 1
    out_w = iw - pw + 1
    scores = np.zeros((out_h, out_w))

    patch_mean = patch.mean()
    patch_zero = patch - patch_mean
    patch_norm = np.sqrt(np.sum(patch_zero ** 2)) + 1e-8

    for y in range(out_h):
        for x in range(out_w):
            window = image[y:y+ph, x:x+pw]
            window_mean = window.mean()
            window_zero = window - window_mean
            window_norm = np.sqrt(np.sum(window_zero ** 2)) + 1e-8

            numerator = np.sum(patch_zero * window_zero)
            scores[y, x] = numerator / (patch_norm * window_norm)

    return scores

#@njit
def find_correspondence(img1, img2, pt, patch_size=21):

    c, r = pt
    r = int(r)
    c = int(c)
    p = patch_size // 2
    patch = img1[r-p:r+p+1, c-p:c+p+1].astype(np.float64)

    scores = ncc_match(patch, img2.astype(np.float64))

    best_flat = np.argmax(scores)

    best_r = best_flat // scores.shape[1] + p
    best_c = best_flat % scores.shape[1] + p

    return (best_r, best_c), scores.max()


def compute_homography(src_pts, dst_pts):

    A = []
    b_vec = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0, -x*xp, -y*xp])
        b_vec.append(xp)
        A.append([0, 0, 0, x, y, 1, -x*yp, -y*yp])
        b_vec.append(yp)

    A = np.array(A)
    b_vec = np.array(b_vec)

    params, residuals, rank, singular_values = np.linalg.lstsq(A, b_vec, rcond=None)
    a, b, c, d, e, f, g, h = params

    if len(residuals) > 0:
        n_points = len(src_pts)
        avg_pixel_error = np.sqrt(residuals[0] / (2 * n_points))
        print(f"average pixel error per coordinate: {avg_pixel_error:.2f} px")

    H = np.array([[a, b, c],
                  [d, e, f],
                  [g, h, 1]])
    return H


def get_warped_corners(img_ht, img_wt, H):

    corners = np.array([[0,0,1], [img_wt,0,1], [0,img_ht,1], [img_wt,img_ht,1]])

    warped_corners = []
    for i in corners:
        m = np.matmul(H, i.T)
        w = m[2]
        warped_corners.append(np.array([m[0]/w, m[1]/w, 1]))

    return np.array(warped_corners)


def compute_canvas_bounds(all_corners):

    all_corners = np.vstack(all_corners)

    min_x = min(i[0] for i in all_corners)
    min_y = min(i[1] for i in all_corners)
    max_x = max(i[0] for i in all_corners)
    max_y = max(i[1] for i in all_corners)

    out_wt = math.ceil(max_x - min_x)
    out_ht = math.ceil(max_y - min_y)

    return out_ht, out_wt, min_x, min_y

#@njit
def warp_inverse_nn(img, H, img_ht, img_wt, out_ht, out_wt, off_x, off_y):

    H_inv = np.linalg.inv(H)
    output_img = np.zeros((out_ht, out_wt), dtype=img.dtype)

    for i in range(out_ht):
        for j in range(out_wt):
            x, y = j + off_x, i + off_y
            raw = np.matmul(H_inv, np.array([x, y, 1]))
            w = raw[2]
            if w <= 1e-6:
                continue
            xp = raw[0]/w
            yp = raw[1]/w
            col_src = int(round(xp))
            row_src = int(round(yp))
            if 0 <= row_src < img_ht and 0 <= col_src < img_wt:
                output_img[i][j] = img[row_src][col_src]

    return output_img

#@njit
def warp_inverse_bilinear(img, H, img_ht, img_wt, out_ht, out_wt, off_x, off_y):

    H_inv = np.linalg.inv(H)
    output_img = np.zeros((out_ht, out_wt), dtype=img.dtype)

    for i in range(out_ht):
        for j in range(out_wt):
            x, y = j + off_x, i + off_y
            raw = np.matmul(H_inv, np.array([x, y, 1]))
            w = raw[2]
            if w <= 1e-6:
                continue
            xp = raw[0]/w
            yp = raw[1]/w

            x0 = int(math.floor(xp))
            y0 = int(math.floor(yp))
            x1_ = x0 + 1
            y1_ = y0 + 1

            if 0 <= x0 < img_wt-1 and 0 <= y0 < img_ht-1:
                dx = xp - x0
                dy = yp - y0

                top = img[y0][x0]*(1-dx) + img[y0][x1_]*dx
                bottom = img[y1_][x0]*(1-dx) + img[y1_][x1_]*dx
                pixel = top*(1-dy) + bottom*dy

                output_img[i][j] = pixel

    return output_img


def place_image(canvas, img, off_x, off_y):

    ht, wt = img.shape[:2]
    x0 = int(round(-off_x))
    y0 = int(round(-off_y))
    canvas[y0:y0+ht, x0:x0+wt] = img
    return canvas


def composite_images(base_img, warp_img):

    canvas = base_img.copy()
    empty_mask = (base_img == 0)
    canvas[empty_mask] = warp_img[empty_mask]
    return canvas


def compute_overlap_mask(warped_outer, placed_mid):

    mask_outer = (warped_outer != 0)
    mask_mid = (placed_mid != 0)
    overlap_mask = mask_outer & mask_mid

    return overlap_mask


def estimate_intensity_transform(warped_outer, placed_mid, overlap_mask):

    outer_vals = warped_outer[overlap_mask].astype(np.float64)
    mid_vals = placed_mid[overlap_mask].astype(np.float64)

    A = np.stack([outer_vals, np.ones_like(outer_vals)], axis=1)
    params, residuals, rank, sv = np.linalg.lstsq(A, mid_vals, rcond=None)
    a, b = params

    return a, b


def apply_intensity_transform(img, a, b):

    valid_mask = (img != 0)
    corrected = a * img.astype(np.float64) + b
    corrected = np.clip(corrected, 0, 255).astype(img.dtype)
    corrected[~valid_mask] = 0

    return corrected


def apply_artificial_exposure(img, gain, bias):

    altered = img.astype(np.float64) * gain + bias
    altered = np.clip(altered, 0, 255).astype(img.dtype)

    return altered


def run_pipeline(imgL_g, imgM_g, imgR_g, left_pts, right_pts, label):

    imgM_correspondencePointsL = [find_correspondence(imgL_g, imgM_g, pt) for pt in left_pts]
    imgM_correspondencePointsR = [find_correspondence(imgR_g, imgM_g, pt) for pt in right_pts]

    print(f"\n[{label}] correspondences on middle image from left = \n", imgM_correspondencePointsL)
    print(f"\n[{label}] correspondences on middle image from right = \n", imgM_correspondencePointsR)

    def plot_correspondences(img_gray, correspondences, title, fname):
        plt.imshow(img_gray, cmap='gray')
        for (r, c), score in correspondences:
            plt.scatter(c, r, c='red', marker='+', s=100)
        plt.title(title)
        plt.savefig(f"{out_dir}/{fname}", dpi=150)
        plt.show()

    plot_correspondences(imgM_g, imgM_correspondencePointsL,
                          f"{label}: found points from left on middle",
                          f"{label}_found_points_left_on_mid.png")
    plot_correspondences(imgM_g, imgM_correspondencePointsR,
                          f"{label}: found points from right on middle",
                          f"{label}_found_points_right_on_mid.png")

    src_ptsL = list(left_pts)
    dst_ptsL = [(c, r) for (r, c), score in imgM_correspondencePointsL]

    src_ptsR = list(right_pts)
    dst_ptsR = [(c, r) for (r, c), score in imgM_correspondencePointsR]

    H_left_to_mid = compute_homography(src_ptsL, dst_ptsL)
    H_right_to_mid = compute_homography(src_ptsR, dst_ptsR)

    print(f"[{label}] homography left to mid =\n", H_left_to_mid)
    print(f"[{label}] homography right to mid =\n", H_right_to_mid)

    left_corners = get_warped_corners(imgL_ht, imgL_wt, H_left_to_mid)
    right_corners = get_warped_corners(imgR_ht, imgR_wt, H_right_to_mid)
    mid_corners = np.array([[0,0,1], [imgM_wt,0,1], [0,imgM_ht,1], [imgM_wt,imgM_ht,1]])

    out_ht, out_wt, off_x, off_y = compute_canvas_bounds([left_corners, right_corners, mid_corners])
    print(f"[{label}] final canvas size (h x w) = {out_ht} x {out_wt}")

    result_nnL = warp_inverse_nn(imgL_g, H_left_to_mid, imgL_ht, imgL_wt, out_ht, out_wt, off_x, off_y)
    result_bilinearL = warp_inverse_bilinear(imgL_g, H_left_to_mid, imgL_ht, imgL_wt, out_ht, out_wt, off_x, off_y)

    result_nnR = warp_inverse_nn(imgR_g, H_right_to_mid, imgR_ht, imgR_wt, out_ht, out_wt, off_x, off_y)
    result_bilinearR = warp_inverse_bilinear(imgR_g, H_right_to_mid, imgR_ht, imgR_wt, out_ht, out_wt, off_x, off_y)

    canvas_nn = np.zeros((out_ht, out_wt), dtype=imgM_g.dtype)
    canvas_bilinear = np.zeros((out_ht, out_wt), dtype=imgM_g.dtype)

    canvas_nn = place_image(canvas_nn, imgM_g, off_x, off_y)
    canvas_bilinear = place_image(canvas_bilinear, imgM_g, off_x, off_y)

    final_nn = composite_images(canvas_nn, result_nnL)
    final_bilinear = composite_images(canvas_bilinear, result_bilinearL)

    final_nn = composite_images(final_nn, result_nnR)
    final_bilinear = composite_images(final_bilinear, result_bilinearR)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(final_nn, cmap='gray')
    axes[0].set_title(f"{label}: panorama nearest neighbour")
    axes[1].imshow(final_bilinear, cmap='gray')
    axes[1].set_title(f"{label}: panorama bilinear")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{label}_panorama.png", dpi=150)
    plt.show()

    overlap_mask_L = compute_overlap_mask(result_nnL, canvas_nn)
    a_L, b_L = estimate_intensity_transform(result_nnL, canvas_nn, overlap_mask_L)
    result_nnL_corrected = apply_intensity_transform(result_nnL, a_L, b_L)
    result_bilinearL_corrected = apply_intensity_transform(result_bilinearL, a_L, b_L)

    overlap_mask_R = compute_overlap_mask(result_nnR, canvas_nn)
    a_R, b_R = estimate_intensity_transform(result_nnR, canvas_nn, overlap_mask_R)
    result_nnR_corrected = apply_intensity_transform(result_nnR, a_R, b_R)
    result_bilinearR_corrected = apply_intensity_transform(result_bilinearR, a_R, b_R)

    print(f"[{label}] left gain/bias (a,b) = ({a_L}, {b_L})")
    print(f"[{label}] right gain/bias (a,b) = ({a_R}, {b_R})")

    final_nn_corrected = composite_images(canvas_nn, result_nnL_corrected)
    final_nn_corrected = composite_images(final_nn_corrected, result_nnR_corrected)

    final_bilinear_corrected = composite_images(canvas_bilinear, result_bilinearL_corrected)
    final_bilinear_corrected = composite_images(final_bilinear_corrected, result_bilinearR_corrected)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(final_nn_corrected, cmap='gray')
    axes[0].set_title(f"{label}: panorama nn, intensity corrected")
    axes[1].imshow(final_bilinear_corrected, cmap='gray')
    axes[1].set_title(f"{label}: panorama bilinear, intensity corrected")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{label}_panorama_corrected.png", dpi=150)
    plt.show()

    diff = np.abs(final_nn.astype(int) - final_nn_corrected.astype(int)).astype(np.uint8)
    plt.imshow(diff, cmap='gray')
    plt.title(f"{label}: difference before vs after correction")
    plt.colorbar()
    plt.savefig(f"{out_dir}/{label}_difference.png", dpi=150)
    plt.show()

    return (a_L, b_L), (a_R, b_R)

plt.imshow(imgL_gray,cmap='gray')
left_keyPoints = plt.ginput(-1,-1,True)
plt.imshow(imgR_gray,cmap='gray')
right_keyPoints = plt.ginput(-1,-1,True)


print("left keypoints (col,row from ginput) = ", left_keyPoints)
print("right keypoints (col,row from ginput) = ", right_keyPoints)

plt.imshow(imgL_gray, cmap='gray')
plt.scatter([p[0] for p in left_keyPoints], [p[1] for p in left_keyPoints], c='red', marker='+', s=100)
plt.title("manually picked points on left image")
plt.savefig(f"{out_dir}/03_left_points_marked.png", dpi=150)
plt.show()

plt.imshow(imgR_gray, cmap='gray')
plt.scatter([p[0] for p in right_keyPoints], [p[1] for p in right_keyPoints], c='red', marker='+', s=100)
plt.title("manually picked points on right image")
plt.savefig(f"{out_dir}/04_right_points_marked.png", dpi=150)
plt.show()


(a_L_natural, b_L_natural), (a_R_natural, b_R_natural) = run_pipeline(
    imgL_gray, imgM_gray, imgR_gray, left_keyPoints, right_keyPoints, label="natural")

gain_L, bias_L = 0.6, -25
gain_R, bias_R = 1.4, 15

imgL_gray_altered = apply_artificial_exposure(imgL_gray, gain_L, bias_L)
imgR_gray_altered = apply_artificial_exposure(imgR_gray, gain_R, bias_R)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(imgL_gray_altered, cmap='gray')
axes[0].set_title("left image, artificially darkened")
axes[1].imshow(imgR_gray_altered, cmap='gray')
axes[1].set_title("right image, artificially brightened")
plt.tight_layout()
plt.savefig(f"{out_dir}/05_altered_left_right_images.png", dpi=150)
plt.show()


(a_L_alt, b_L_alt), (a_R_alt, b_R_alt) = run_pipeline(
    imgL_gray_altered, imgM_gray, imgR_gray_altered, left_keyPoints, right_keyPoints, label="altered")

print("\nsummary")
print(f"natural left gain/bias  = ({a_L_natural:.3f}, {b_L_natural:.3f})")
print(f"natural right gain/bias = ({a_R_natural:.3f}, {b_R_natural:.3f})")
print(f"applied left exposure   = gain {gain_L}, bias {bias_L} -> expected recovered gain approx {1/gain_L:.3f}")
print(f"applied right exposure  = gain {gain_R}, bias {bias_R} -> expected recovered gain approx {1/gain_R:.3f}")
print(f"recovered left gain/bias  (altered case) = ({a_L_alt:.3f}, {b_L_alt:.3f})")
print(f"recovered right gain/bias (altered case) = ({a_R_alt:.3f}, {b_R_alt:.3f})")