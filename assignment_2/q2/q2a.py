import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from q2functions import nearest_neighbor_resize

image = np.array(Image.open("newspaper.jpeg").convert("L"))
scale_factors = [1, 4, 8, 12, 16, 20, 26, 30, 40]

plt.figure(figsize=(14, 9))

for i, factor in enumerate(scale_factors):

    # Factor 1 means no downsampling
    resized_image = nearest_neighbor_resize(image,factor)

    plt.subplot(3, 3, i + 1)

    plt.imshow(resized_image,cmap="gray",interpolation="nearest")

    plt.title(f"Scale factor = {factor}\n" f"Size = {resized_image.shape}")

    plt.axis("off")


plt.tight_layout()
plt.savefig(f"q2a.png", dpi=130, bbox_inches="tight")
plt.show()