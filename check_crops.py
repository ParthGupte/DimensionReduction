import os
import glob
import matplotlib.pyplot as plt
from svd_helper import preprocess_image_coord_crop

IMAGE_DIR = "MESSIDOR/images"
CROP_SIZE = 512
DEVICE = "cpu"
NUM_EXAMPLES = 6
OUTPUT_PATH = "crop_check.png"

image_files = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    + glob.glob(os.path.join(IMAGE_DIR, "*.JPG"))
)

examples = []
for f in image_files:
    crops = preprocess_image_coord_crop(f, CROP_SIZE, DEVICE)
    if crops is None:
        continue
    examples.append((f, crops))
    if len(examples) == NUM_EXAMPLES:
        break

print(f"Found {len(examples)} examples with valid OD/FOVEA coordinates.")

fig, axes = plt.subplots(len(examples), 2, figsize=(6, 3 * len(examples)))
if len(examples) == 1:
    axes = axes[None, :]

for row, (f, crops) in enumerate(examples):
    od_img = crops["OD"].reshape(3, CROP_SIZE, CROP_SIZE).permute(1, 2, 0).cpu().numpy()
    fovea_img = crops["FOVEA"].reshape(3, CROP_SIZE, CROP_SIZE).permute(1, 2, 0).cpu().numpy()

    axes[row, 0].imshow(od_img)
    axes[row, 0].set_title(f"{os.path.basename(f)} - OD")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(fovea_img)
    axes[row, 1].set_title(f"{os.path.basename(f)} - FOVEA")
    axes[row, 1].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"Saved crop check grid to {OUTPUT_PATH}")
