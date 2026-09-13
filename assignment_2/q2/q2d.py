import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter
from q2functions import create_notch_filter, optimum_notch_restore

image = np.array(Image.open("newspaper.jpeg").convert("L")).astype(np.float64)
g = image  # alias matching the textbook notation: g(x,y) = noisy image

F = np.fft.fft2(g)
F_shifted = np.fft.fftshift(F)

notch_points = [
    # inner ring (fundamental frequencies)
    (832, 718), (766, 774),
    (769, 716), (829, 776),
    (802, 688), (796, 804),
    (863, 749), (735, 743),
    # outer ring (2nd-order harmonics of the same lattice)
    (740, 686), (858, 806),
    (735, 804), (863, 688),
    (827, 835), (771, 657),
    (764, 833), (834, 659),
    (704, 713), (894, 779),
    (702, 773), (896, 719),
]
radius = 8

# Notch-REJECT filter (0 at the noise frequencies, 1 elsewhere) -- same helper as part 2
H_reject = create_notch_filter(g.shape, notch_points, radius)

# The complementary notch-PASS filter keeps ONLY the small neighbourhoods
# around the noise frequencies and kills everything else.
H_pass = 1 - H_reject

# 3. Estimate the interference/noise pattern eta(x,y)
#    eta = IFT{ H_pass(u,v) * G(u,v) }
N_shifted = H_pass * F_shifted
eta = np.real(np.fft.ifft2(np.fft.ifftshift(N_shifted)))


windows = [7,15,31,63]  # (2a+1) x (2b+1) neighbourhood, a = b = 15  -- justified in the discussion
optimum_notch_restore_results=[]
for i in windows:
    w, modulated_noise, restored = optimum_notch_restore(g, eta, i)
    optimum_notch_restore_results.append([w,modulated_noise,restored])

# 5. Required plots: estimated noise, weight function,
#    modulated noise, restored image
fig, axes = plt.subplots(4, 4, figsize=(14, 14))

for i in range(4):
    w = optimum_notch_restore_results[i][0]
    modulated_noise = optimum_notch_restore_results[i][1]
    restored = optimum_notch_restore_results[i][2]
    win = windows[i]
    
    axes[i, 0].imshow(eta, cmap="gray")
    axes[i, 0].set_title(r"Estimated noise $\eta(x,y)$")
    axes[i, 0].axis("off")

    wlo, whi = np.percentile(w, [1, 99])
    axes[i, 1].imshow(w, cmap="gray", vmin=wlo, vmax=whi)
    axes[i, 1].set_title(f"Weight function $w(x,y)$  (window={win}x{win})")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(modulated_noise, cmap="gray")
    axes[i, 2].set_title(r"Modulated noise $w(x,y)\,\eta(x,y)$")
    axes[i, 2].axis("off")

    axes[i, 3].imshow(restored, cmap="gray")
    axes[i, 3].set_title("Restored image  " + r"$\hat f = g - w\eta$")
    axes[i, 3].axis("off")

plt.tight_layout()
plt.savefig(f"q2d.png", dpi=130, bbox_inches="tight")
plt.show()
plt.close()

print("eta   min/max:", eta.min(), eta.max())
print("w     min/max:", w.min(), w.max())
print("wta    min/max:", modulated_noise.min(), modulated_noise.max())
print("rest  min/max:", restored.min(), restored.max())
