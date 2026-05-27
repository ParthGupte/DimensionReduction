import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from svd_helper import *

# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = "MESSIDOR/images"

SAVE_DIR_RAW = "reconstructed_raw"
SAVE_DIR_COV = "reconstructed_cov"
SAVE_DIR_COR = "reconstructed_cor"

os.makedirs(SAVE_DIR_RAW, exist_ok=True)
os.makedirs(SAVE_DIR_COV, exist_ok=True)
os.makedirs(SAVE_DIR_COR, exist_ok=True)

DEVICE = "cuda:1"

CALC_BATCH_SIZE = 10

BETA = 0.01

# ============================================================
# LOAD SAVED MATRICES
# ============================================================

HTH = torch.load(
    "saved_matrices/HTH.pt",
    map_location=DEVICE
)

mu = torch.load(
    "saved_matrices/mu.pt",
    map_location=DEVICE
).unsqueeze(1)

sigma = torch.load(
    "saved_matrices/sigma.pt",
    map_location=DEVICE
)

sigma_inv = 1.0 / sigma

sigma_inv[torch.isinf(sigma_inv)] = 0

n = HTH.shape[0]
d = mu.shape[0]

print("n =", n)
print("d =", d)

# ============================================================
# BUILD COVARIANCE / CORRELATION MATRICES
# ============================================================

one = torch.ones((n, 1), device=DEVICE)

HTH_cov = (
    HTH
    + (one @ mu.T @ mu @ one.T)
    - (one @ one.T @ HTH / n)
    - (HTH / n @ one @ one.T)
) / n

HTH_cor = (
    sigma_inv[:, None]
    * HTH_cov
    * sigma_inv[None, :]
)

# ============================================================
# EIGENDECOMPOSITION
# ============================================================

print("Computing eigendecomposition...")

eigenvalues_raw, eigenvectors_raw = torch.linalg.eigh(HTH)

eigenvalues_cov, eigenvectors_cov = torch.linalg.eigh(HTH_cov)

eigenvalues_cor, eigenvectors_cor = torch.linalg.eigh(HTH_cor)

# ============================================================
# SORT DESCENDING
# ============================================================

idx_raw = torch.argsort(
    eigenvalues_raw,
    descending=True
)

idx_cov = torch.argsort(
    eigenvalues_cov,
    descending=True
)

idx_cor = torch.argsort(
    eigenvalues_cor,
    descending=True
)

eigenvalues_raw = eigenvalues_raw[idx_raw]
eigenvectors_raw = eigenvectors_raw[:, idx_raw]

eigenvalues_cov = eigenvalues_cov[idx_cov]
eigenvectors_cov = eigenvectors_cov[:, idx_cov]

eigenvalues_cor = eigenvalues_cor[idx_cor]
eigenvectors_cor = eigenvectors_cor[:, idx_cor]

# ============================================================
# PICK K
# ============================================================

k_raw, _, _, _ = analyze_precomputed_eigenvalues(
    eigenvalues_raw,
    title_suffix="RawMoments",
    beta=BETA
)

k_cov, _, _, _ = analyze_precomputed_eigenvalues(
    eigenvalues_cov,
    title_suffix="Covariance",
    beta=BETA
)

k_cor, _, _, _ = analyze_precomputed_eigenvalues(
    eigenvalues_cor,
    title_suffix="Correlation",
    beta=BETA
)

print("k_raw =", k_raw)
print("k_cov =", k_cov)
print("k_cor =", k_cor)

# ============================================================
# TOP-K EIGENVECTORS
# ============================================================

V_raw = eigenvectors_raw[:, :k_raw]

V_cov = eigenvectors_cov[:, :k_cov]

V_cor = eigenvectors_cor[:, :k_cor]

# ============================================================
# IMAGE FILES
# ============================================================

image_files = glob.glob(
    os.path.join(IMAGE_DIR, '*.png')
) + glob.glob(
    os.path.join(IMAGE_DIR, '*.jpg')
) + glob.glob(
    os.path.join(IMAGE_DIR, '*.JPG')
)

image_files.sort()

print(f"Found {len(image_files)} images")

# ============================================================
# IMAGE SIZE
# ============================================================

size_counter, failed_images = count_image_sizes(IMAGE_DIR)

print("Image size distribution:\n")

for (w, h), count in sorted(size_counter.items()):
    print(f"{w} x {h} : {count} images")

MAX_SIZE = w

# ============================================================
# COMPUTE LATENT MATRICES
# ============================================================

print("Computing latent matrices...")

Z_raw = torch.zeros(
    (d, k_raw),
    device=DEVICE
)

Z_cov = torch.zeros(
    (d, k_cov),
    device=DEVICE
)

Z_cor = torch.zeros(
    (d, k_cor),
    device=DEVICE
)

for start in tqdm(range(0, n, CALC_BATCH_SIZE)):

    end = min(start + CALC_BATCH_SIZE, n)

    # --------------------------------------------------------
    # LOAD BATCH
    # --------------------------------------------------------

    batch_vecs = []

    for i in range(start, end):

        vec = preprocess_image_noresize(
            image_files[i],
            MAX_SIZE,
            DEVICE,
            i
        )
        # break

        batch_vecs.append(vec)

    H_batch = torch.stack(
        batch_vecs,
        dim=1
    )  # (d, batch)

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------

    Z_raw += (
        H_batch
        @ V_raw[start:end]
    )

    # --------------------------------------------------------
    # COVARIANCE
    # --------------------------------------------------------

    H_centered = H_batch - mu

    Z_cov += (
        H_centered
        @ V_cov[start:end]
    )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    sigma_batch_inv = sigma_inv[start:end]

    H_standardized = (
        H_centered
        * sigma_batch_inv.unsqueeze(0)
    )

    Z_cor += (
        H_standardized
        @ V_cor[start:end]
    )

# ============================================================
# RECONSTRUCT IMAGES (BATCHED)
# ============================================================

print("Reconstructing images...")

mu_vec = mu.squeeze(1)

for start in tqdm(range(0, n, CALC_BATCH_SIZE)):

    end = min(start + CALC_BATCH_SIZE, n)

    batch_size = end - start

    # ========================================================
    # GET BATCH EIGENVECTORS
    # ========================================================

    V_raw_batch = V_raw[start:end].T      # (k_raw, batch)

    V_cov_batch = V_cov[start:end].T      # (k_cov, batch)

    V_cor_batch = V_cor[start:end].T      # (k_cor, batch)

    # ========================================================
    # RAW RECONSTRUCTION
    # ========================================================

    H_recon_raw = (
        Z_raw @ V_raw_batch
    )                                       # (d, batch)

    # ========================================================
    # COVARIANCE RECONSTRUCTION
    # ========================================================

    H_recon_cov = (
        Z_cov @ V_cov_batch
    ) + mu                                  # (d, batch)

    # ========================================================
    # CORRELATION RECONSTRUCTION
    # ========================================================

    H_recon_cor = (
        Z_cor @ V_cor_batch
    )                                       # (d, batch)

    sigma_batch = sigma[start:end]          # (batch,)

    H_recon_cor = (
        H_recon_cor
        * sigma_batch.unsqueeze(0)
        + mu
    )

    # ========================================================
    # CLAMP
    # ========================================================

    # ========================================================
    # MOVE TO CPU ONCE
    # ========================================================

    H_recon_raw = (
        H_recon_raw
        .mul(255)
        .clamp(0,255)
        .byte()
        .cpu()
        )

    H_recon_cov = (
        H_recon_cov
        .mul(255)
        .clamp(0,255)
        .byte()
        .cpu()
        )

    H_recon_cor = (
        H_recon_cor
        .mul(255)
        .clamp(0,255)
        .byte()
        .cpu()
        )

    # ========================================================
    # SAVE EACH IMAGE
    # ========================================================

    for b in range(batch_size):

        idx = start + b

        # ----------------------------------------------------
        # RAW
        # ----------------------------------------------------

        img_raw = (
            H_recon_raw[:, b]
            .reshape(3, MAX_SIZE, MAX_SIZE)
            .permute(1,2,0)
            .numpy()
        )

        Image.fromarray(img_raw).save(
            os.path.join(
                SAVE_DIR_RAW,
                f"recon_raw_{idx:05d}.png"
            )
        )

        # ----------------------------------------------------
        # COV
        # ----------------------------------------------------

        img_cov = (
            H_recon_cov[:, b]
            .reshape(3, MAX_SIZE, MAX_SIZE)
            .permute(1,2,0)
            .numpy()
        )

        Image.fromarray(img_cov).save(
            os.path.join(
                SAVE_DIR_COV,
                f"recon_cov_{idx:05d}.png"
            )
        )

        # ----------------------------------------------------
        # COR
        # ----------------------------------------------------

        img_cor = (
            H_recon_cor[:, b]
            .reshape(3, MAX_SIZE, MAX_SIZE)
            .permute(1,2,0)
            .numpy()
        )

        Image.fromarray(img_cor).save(
            os.path.join(
                SAVE_DIR_COR,
                f"recon_cor_{idx:05d}.png"
            )
        )

print("Done.")