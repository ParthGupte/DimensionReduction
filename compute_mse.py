import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import re

# ============================================================
# CONFIG
# ============================================================

ORIGINAL_DIR = "original_images"

# ------------------------------------------------------------
# Structure:
#
# method -> beta -> folder
# ------------------------------------------------------------

RECON_DIRS = {

    "raw": {
        0.01: "reconstructed_raw",
        0.05: "reconstructed_raw_95",
        0.10: "reconstructed_raw_90",
    },

    "cov": {
        0.01: "reconstructed_cov",
        0.05: "reconstructed_cov_95",
        0.10: "reconstructed_cov_90",
    },

    "cor": {
        0.01: "reconstructed_cor",
        0.05: "reconstructed_cor_95",
        0.10: "reconstructed_cor_90",
    },
}

VALID_EXTENSIONS = (
    "*.png",
    "*.jpg",
    "*.JPG",
    "*.jpeg",
)


# ============================================================
# HELPERS
# ============================================================

def extract_image_id(path):
    """
    Extract image ID and normalize it.

    Examples:
        image_0.png            -> 00000
        image_12.png           -> 00012
        recon_raw_00012.png   -> 00012
    """

    filename = os.path.splitext(
        os.path.basename(path)
    )[0]

    matches = re.findall(r"\d+", filename)

    if len(matches) == 0:
        return None

    # take final numeric block
    img_id = int(matches[-1])

    # normalize to 5 digits
    return f"{img_id:05d}"


def load_image(path):
    """
    Load image as float32 RGB in [0,1].
    """

    img = Image.open(path).convert("RGB")

    img = np.asarray(img).astype(np.float32) / 255.0

    return img


def compute_mse(img1, img2):
    """
    Pixel-wise MSE.
    """

    return np.mean((img1 - img2) ** 2)


# ============================================================
# BUILD ORIGINAL IMAGE MAP
# ============================================================

print("Indexing original images...")

original_files = []

for ext in VALID_EXTENSIONS:

    original_files.extend(
        glob.glob(os.path.join(ORIGINAL_DIR, ext))
    )

original_map = {}

for path in original_files:

    img_id = extract_image_id(path)

    original_map[img_id] = path

print(f"Found {len(original_map)} original images")


# ============================================================
# EVALUATE
# ============================================================

results = {}

for method, beta_dirs in RECON_DIRS.items():

    results[method] = {}

    for beta, recon_dir in beta_dirs.items():

        print(f"\nEvaluating {method} | beta={beta}")

        recon_files = []

        for ext in VALID_EXTENSIONS:

            recon_files.extend(
                glob.glob(os.path.join(recon_dir, ext))
            )

        total_mse = 0.0

        count = 0

        missing = 0

        for recon_path in tqdm(recon_files):

            img_id = extract_image_id(recon_path)

            if img_id not in original_map:

                missing += 1

                continue

            original_path = original_map[img_id]

            # ------------------------------------------------
            # LOAD
            # ------------------------------------------------

            recon_img = load_image(recon_path)

            original_img = load_image(original_path)

            # ------------------------------------------------
            # SHAPE CHECK
            # ------------------------------------------------

            if recon_img.shape != original_img.shape:

                print(
                    f"Shape mismatch for ID {img_id}: "
                    f"{recon_img.shape} vs {original_img.shape}"
                )

                continue

            # ------------------------------------------------
            # MSE
            # ------------------------------------------------

            mse = compute_mse(
                recon_img,
                original_img
            )

            total_mse += mse

            count += 1

        avg_mse = (
            total_mse / count
            if count > 0 else float("nan")
        )

        results[method][beta] = {
            "avg_mse": avg_mse,
            "num_images": count,
            "missing_matches": missing,
        }


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n============================================================")
print("RESULTS")
print("============================================================")

for method in results:

    print(f"\nMETHOD: {method}")

    for beta in sorted(results[method].keys()):

        stats = results[method][beta]

        print(
            f"beta={beta:<5} | "
            f"MSE={stats['avg_mse']:.8f} | "
            f"N={stats['num_images']}"
        )


# ============================================================
# LATEX TABLE GENERATOR
# ============================================================

def generate_latex_table(results):
    """
    Generates LaTeX table with subtables for:
        raw / cov / cor

    Columns are beta values.
    """

    methods = ["raw", "cov", "cor"]

    latex = []

    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\small")

    for method in methods:

        betas = sorted(results[method].keys())

        latex.append(r"\begin{subtable}{0.32\linewidth}")
        latex.append(r"\centering")

        latex.append(
            rf"\caption{{{method.upper()} Reconstruction}}"
        )

        col_format = "c" * (len(betas) + 1)

        latex.append(
            rf"\begin{{tabular}}{{{col_format}}}"
        )

        latex.append(r"\hline")

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = ["Metric"]

        for beta in betas:

            header.append(rf"$\beta={beta}$")

        latex.append(
            " & ".join(header) + r" \\"
        )

        latex.append(r"\hline")

        # ----------------------------------------------------
        # MSE ROW
        # ----------------------------------------------------

        mse_row = ["MSE"]

        for beta in betas:

            mse = results[method][beta]["avg_mse"]

            mse_row.append(f"{mse:.6f}")

        latex.append(
            " & ".join(mse_row) + r" \\"
        )

        latex.append(r"\hline")

        latex.append(r"\end{tabular}")

        latex.append(r"\end{subtable}")

        latex.append(r"\hfill")

    latex.append(
        r"\caption{Average reconstruction MSE for different values of $\beta$.}"
    )

    latex.append(
        r"\label{tab:reconstruction_mse}"
    )

    latex.append(r"\end{table*}")

    return "\n".join(latex)


# ============================================================
# PRINT LATEX TABLE
# ============================================================

latex_table = generate_latex_table(results)

print("\n============================================================")
print("LATEX TABLE")
print("============================================================\n")

print(latex_table)