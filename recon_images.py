import os
import csv
import glob
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

from svd_helper import *

# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = "MESSIDOR/images"
MATRIX_DIR = "saved_matrices"
COORDS_CSV = "messidor_od_fovea.csv"

CROP_CONFIGS = [
    ("OD", 256),
    ("OD", 512),
    ("FOVEA", 256),
    ("FOVEA", 512),
]

VARIANTS = ["raw", "cov", "cor"]

BETA_LST = [0.01, 0.05, 0.10]

BETA_SUFFIX = {
    0.01: "",
    0.05: "_95",
    0.10: "_90",
}

N_SAMPLES = 5

SAVE_IMAGES = True

DEVICE = "cuda:0"

CALC_BATCH_SIZE = 10

METRICS_CSV = "crop_reconstruction_mse.csv"


# ============================================================
# HELPERS
# ============================================================

def load_image_files():
    """
    Same glob + sort + coordinate-filter as dim-red-crops.py, so index i
    matches the row/col i used when the crop config's HTH/mu/sigma were built.
    """

    image_files = (
        glob.glob(os.path.join(IMAGE_DIR, '*.png'))
        + glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
        + glob.glob(os.path.join(IMAGE_DIR, '*.JPG'))
    )
    image_files.sort()

    coords_df = pd.read_csv(COORDS_CSV)
    valid_image_ids = set(
        coords_df.dropna(subset=["od_x", "od_y", "fovea_x", "fovea_y"])["image_id"]
    )

    return [
        f for f in image_files
        if os.path.splitext(os.path.basename(f))[0] in valid_image_ids
    ]


def eigenvalues_from_eigenvectors(HTH_variant, eigenvectors):
    """
    Recovers eigenvalues in `eigenvectors`' column order via the Rayleigh
    quotient (v_i^T HTH_variant v_i = lambda_i for an orthonormal eigenvector
    v_i), instead of re-running torch.linalg.eigh.
    """

    HTH_v_V = HTH_variant @ eigenvectors
    return (eigenvectors * HTH_v_V).sum(dim=0)


def k_for_betas(eigenvalues_sorted, beta_lst):
    """
    Same k-selection arithmetic as analyze_precomputed_eigenvalues
    (svd_helper.py), inlined so it doesn't rewrite the existing CSVs/plots.
    """

    variance = eigenvalues_sorted.clamp(min=0)
    total_variance = variance.sum()
    cumulative = torch.cumsum(variance / total_variance, dim=0)

    k_by_beta = {}

    for beta in beta_lst:
        target = torch.tensor(1 - beta, device=cumulative.device, dtype=cumulative.dtype)
        k = int(torch.searchsorted(cumulative, target).item()) + 1
        k_by_beta[beta] = min(k, eigenvalues_sorted.shape[0])

    return k_by_beta


def save_crop_png(vec, size, path):
    img = (
        vec.reshape(3, size, size)
        .permute(1, 2, 0)
        .mul(255)
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )
    Image.fromarray(img).save(path)


# ============================================================
# PER-CROP-CONFIG PROCESSING
# ============================================================

def process_crop_config(region, size):

    run_name = f"{region}_{size}"
    eigen_dir = os.path.join(MATRIX_DIR, run_name)

    print(f"\n==================== {run_name} ====================")

    # --------------------------------------------------------
    # LOAD ALREADY-SAVED MATRICES / EIGENVECTORS
    # (no torch.linalg.eigh, no analyze_precomputed_eigenvalues here --
    # that analysis already ran in eigen_analysis.py for this run_name)
    # --------------------------------------------------------

    HTH = torch.load(
        os.path.join(MATRIX_DIR, f"HTH_{run_name}.pt"),
        map_location=DEVICE,
    )

    mu = torch.load(
        os.path.join(MATRIX_DIR, f"mu_{run_name}.pt"),
        map_location=DEVICE,
    ).unsqueeze(1)

    sigma = torch.load(
        os.path.join(MATRIX_DIR, f"sigma_{run_name}.pt"),
        map_location=DEVICE,
    )

    sigma_inv = 1.0 / sigma
    sigma_inv[torch.isinf(sigma_inv)] = 0

    n = HTH.shape[0]
    d = mu.shape[0]

    print("n =", n)
    print("d =", d)

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

    HTH_variants = {"raw": HTH, "cov": HTH_cov, "cor": HTH_cor}

    V_sorted = {}
    k_by_beta = {}

    for variant in VARIANTS:

        eigenvectors = torch.load(
            os.path.join(eigen_dir, f"eigenvectors_{variant}.pt"),
            map_location=DEVICE,
        )

        eigenvalues = eigenvalues_from_eigenvectors(HTH_variants[variant], eigenvectors)

        idx = torch.argsort(eigenvalues, descending=True)

        V_sorted[variant] = eigenvectors[:, idx]
        k_by_beta[variant] = k_for_betas(eigenvalues[idx], BETA_LST)

        print(f"{variant}: k(beta) = {k_by_beta[variant]}")

    # --------------------------------------------------------
    # IMAGE FILES
    # --------------------------------------------------------

    image_files = load_image_files()

    assert len(image_files) == n, (
        f"{run_name}: found {len(image_files)} images with coordinates, "
        f"expected {n} to match the saved matrices"
    )

    sample_indices = set(range(min(N_SAMPLES, n)))

    original_dir = f"original_{run_name}"

    if SAVE_IMAGES:
        os.makedirs(original_dir, exist_ok=True)

        for variant in VARIANTS:
            for beta in BETA_LST:
                out_dir = f"reconstructed_{run_name}_{variant}{BETA_SUFFIX[beta]}"
                os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # COMPUTE LATENT MATRICES (single pass, k_max components per variant)
    # --------------------------------------------------------

    k_max = {variant: max(k_by_beta[variant].values()) for variant in VARIANTS}

    Z = {
        variant: torch.zeros((d, k_max[variant]), device=DEVICE)
        for variant in VARIANTS
    }

    print("Computing latent matrices...")

    for start in tqdm(range(0, n, CALC_BATCH_SIZE)):

        end = min(start + CALC_BATCH_SIZE, n)

        batch_vecs = [
            preprocess_image_coord_crop(image_files[i], size, DEVICE)[region]
            for i in range(start, end)
        ]

        H_batch = torch.stack(batch_vecs, dim=1)  # (d, batch)

        Z["raw"] += H_batch @ V_sorted["raw"][start:end, :k_max["raw"]]

        H_centered = H_batch - mu

        Z["cov"] += H_centered @ V_sorted["cov"][start:end, :k_max["cov"]]

        sigma_batch_inv = sigma_inv[start:end]
        H_standardized = H_centered * sigma_batch_inv.unsqueeze(0)

        Z["cor"] += H_standardized @ V_sorted["cor"][start:end, :k_max["cor"]]

        if SAVE_IMAGES:
            for b in range(end - start):
                idx = start + b
                if idx in sample_indices:
                    save_crop_png(
                        H_batch[:, b],
                        size,
                        os.path.join(original_dir, f"orig_{idx:05d}.png"),
                    )

    # --------------------------------------------------------
    # RECONSTRUCT + MSE (per variant, per beta)
    # --------------------------------------------------------

    print("Reconstructing & computing MSE...")

    sq_err_sum = {variant: {beta: 0.0 for beta in BETA_LST} for variant in VARIANTS}

    for start in tqdm(range(0, n, CALC_BATCH_SIZE)):

        end = min(start + CALC_BATCH_SIZE, n)
        batch_size = end - start

        # ground-truth crop batch, in [0, 1] -- reloaded here (rather than
        # kept from the pass above) to avoid holding all n crops in memory
        batch_vecs = [
            preprocess_image_coord_crop(image_files[i], size, DEVICE)[region]
            for i in range(start, end)
        ]

        H_batch = torch.stack(batch_vecs, dim=1)  # (d, batch)

        for variant in VARIANTS:

            V_batch_full = V_sorted[variant][start:end]  # (batch, k_max)

            for beta in BETA_LST:

                k = k_by_beta[variant][beta]

                H_recon = (
                    Z[variant][:, :k]
                    @ V_batch_full[:, :k].T
                )  # (d, batch)

                if variant == "cov":
                    H_recon = H_recon + mu
                elif variant == "cor":
                    sigma_batch = sigma[start:end]
                    H_recon = H_recon * sigma_batch.unsqueeze(0) + mu

                sq_err_sum[variant][beta] += (
                    (H_recon - H_batch).pow(2).sum().item()
                )

                if SAVE_IMAGES:
                    for b in range(batch_size):
                        idx = start + b
                        if idx in sample_indices:
                            out_dir = f"reconstructed_{run_name}_{variant}{BETA_SUFFIX[beta]}"
                            save_crop_png(
                                H_recon[:, b],
                                size,
                                os.path.join(out_dir, f"recon_{variant}_{idx:05d}.png"),
                            )

    # --------------------------------------------------------
    # COLLECT RESULTS
    # --------------------------------------------------------

    rows = []

    for variant in VARIANTS:
        for beta in BETA_LST:

            avg_mse = sq_err_sum[variant][beta] / (n * d)

            rows.append({
                "region": region,
                "size": size,
                "variant": variant,
                "beta": beta,
                "k": k_by_beta[variant][beta],
                "avg_mse": avg_mse,
                "n": n,
                "d": d,
            })

            print(
                f"{run_name} | {variant} | beta={beta} | "
                f"k={k_by_beta[variant][beta]} | avg_mse={avg_mse:.8f}"
            )

    return rows


# ============================================================
# MAIN
# ============================================================

all_results = []

for region, size in CROP_CONFIGS:
    all_results.extend(process_crop_config(region, size))

with open(METRICS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["region", "size", "variant", "beta", "k", "avg_mse", "n", "d"],
    )
    writer.writeheader()
    writer.writerows(all_results)

print(f"\nSaved metrics to {METRICS_CSV}")
print("Done.")
