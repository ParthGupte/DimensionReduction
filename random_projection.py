import numpy as np
import os
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

#check images size
import os
from collections import Counter
from PIL import Image
from svd_helper import *

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


IMAGE_DIR = "MESSIDOR/images"

DEVICE = "cuda:0"

CALC_BATCH_SIZE = 100

MAX_DIM = 6324912

size_counter, failed_images = count_image_sizes(IMAGE_DIR)
print("Image size distribution:\n")
for (w, h), count in sorted(size_counter.items()):
    print(f"{w} x {h} : {count} images")

MAX_SIZE = w
if failed_images:
    print("\nFailed to read the following files:")
    for f in failed_images:
        print(f)

# Load all images
image_files = glob.glob(os.path.join(IMAGE_DIR, '*.png')) + \
              glob.glob(os.path.join(IMAGE_DIR, '*.jpg')) + \
              glob.glob(os.path.join(IMAGE_DIR, '*.JPG'))
image_files.sort()

n = len(image_files)

print(f"Found {n} images.")

def compute_projection_matrix(
    image_files,
    k,
    max_dim,
    batch_size,
    max_size,
    device,
):
    # Gaussian random matrix
    R = torch.randn(max_dim, k, device=device) / np.sqrt(k)

    # Normalize each column to unit norm
    # R = R * np.sqrt(MAX_DIM) / torch.norm(R, dim=0, keepdim=True)

    reduced_features = []
    batch = []

    for f in tqdm(image_files):
        vec = preprocess_image_noresize(f, max_size, device)
        batch.append(vec)

        if len(batch) == batch_size:
            X = torch.stack(batch)
            Y = X @ R

            reduced_features.append(Y.cpu())
            batch = []

    if batch:
        X = torch.stack(batch)
        Y = X @ R
        reduced_features.append(Y.cpu())

    return torch.cat(reduced_features, dim=0)

def pairwise_distance_error(
    image_files,
    reduced_features,
    n_samples,
    max_size,
    device,
):
    errors = []

    N = len(image_files)

    for _ in tqdm(range(n_samples)):

        i, j = np.random.choice(N, 2, replace=False)

        xi = preprocess_image_noresize(
            image_files[i],
            max_size,
            device,
        )

        xj = preprocess_image_noresize(
            image_files[j],
            max_size,
            device,
        )

        d_orig = torch.norm(xi - xj).item()

        yi = reduced_features[i]
        yj = reduced_features[j]

        d_proj = torch.norm(yi - yj).item()

        error = d_proj - d_orig

        errors.append(error)

    return np.mean(errors), np.std(errors)

k_values = [
    10,
    20,
    50,
    70,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    800,
]

mean_errors = []
std_errors = []

for k in k_values:

    print(f"\nRunning k={k}")

    Y = compute_projection_matrix(
        image_files,
        k,
        MAX_DIM,
        CALC_BATCH_SIZE,
        MAX_SIZE,
        DEVICE,
    )

    mean_err, std_err = pairwise_distance_error(
        image_files,
        Y,
        n_samples=1000,
        max_size=MAX_SIZE,
        device=DEVICE,
    )

    mean_errors.append(mean_err)
    std_errors.append(std_err)

    print(
        f"k={k}, mean={mean_err:.4f}, std={std_err:.4f}"
    )

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    mean_errors,
    marker="o",
)

plt.xlabel("k")
plt.ylabel("Mean Pairwise Distance Error")
plt.title("Error vs Projection Dimension")
plt.grid(True)

# Uncomment if k spans powers of 2
# plt.xscale("log", base=2)

plt.tight_layout()
plt.savefig("error_vs_dim.png")