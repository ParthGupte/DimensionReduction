import os
import torch
from svd_helper import *

MATRIX_DIR = "saved_matrices"

BETA_LST = [0.01,0.05,0.1,0.15,0.2]

def eigen_analysis(run_name, beta_lst=BETA_LST):
    """
    Runs eigen-decomposition (raw moments, covariance, correlation) on the
    HTH/mu/sigma matrices saved for `run_name` by dim-red-crops.py, and stores
    every resulting artifact (eigenvectors, projections Z, eigenvalue
    CSVs/plots) under saved_matrices/<run_name>/ so results from different
    runs (e.g. different IMAGE_REGION/CROP_SIZE combinations) never collide.
    """

    output_dir = os.path.join(MATRIX_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    HTH = torch.load(os.path.join(MATRIX_DIR, f"HTH_{run_name}.pt")).to("cpu")
    mu = torch.load(os.path.join(MATRIX_DIR, f"mu_{run_name}.pt")).unsqueeze(1).to("cpu")
    sigma = torch.load(os.path.join(MATRIX_DIR, f"sigma_{run_name}.pt")).to("cpu")

    sigma_inv = 1 / sigma

    device = mu.device

    print(run_name, HTH.shape, mu.shape)
    n = int(HTH.shape[0])
    d = int(mu.shape[0])
    print(n, d)

    one = torch.ones((n, 1), device=device)
    HTH_cov = (HTH + (one @ mu.T @ mu @ one.T) - (one @ one.T @ HTH / n) - (HTH / n @ one @ one.T)) / n
    HTH_cor = sigma_inv[:, None] * HTH_cov * sigma_inv[None, :]

    def run_variant(HTH_variant, variant_key, title_suffix):
        eigenvalues, eigenvectors = torch.linalg.eigh(HTH_variant)
        torch.save(eigenvectors, os.path.join(output_dir, f"eigenvectors_{variant_key}.pt"))

        k, _, _, _ = analyze_precomputed_eigenvalues(
            eigenvalues, title_suffix=title_suffix, output_dir=output_dir
        )
        beta_vs_fro_norm(eigenvalues, beta_lst, title_suffix=title_suffix, output_dir=output_dir)

        Z = compute_projection_Z(eigenvalues, eigenvectors, k)
        torch.save(Z, os.path.join(output_dir, f"Z_{variant_key}.pt"))

    run_variant(HTH, "raw", "RawMoments")
    run_variant(HTH_cov, "cov", "Covariance")
    run_variant(HTH_cor, "cor", "Correlation")


if __name__ == "__main__":
    IMAGE_REGION = "OD"
    CROP_SIZE = 512

    eigen_analysis(f"{IMAGE_REGION}_{CROP_SIZE}")
