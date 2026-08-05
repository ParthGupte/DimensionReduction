import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.io import read_image
import pandas as pd

#check images size
import os
from collections import Counter
from PIL import Image

def count_image_sizes(image_dir):
    size_counter = Counter()
    failed_images = []

    for root, _, files in os.walk(image_dir):
        for fname in files:
            path = os.path.join(root, fname)
            try:
                with Image.open(path) as img:
                    size_counter[img.size] += 1  # (width, height)
            except Exception as e:
                failed_images.append(path)

    return size_counter, failed_images

def show_padded_image(img_padded, title="Padded Image"):

    """
    Displays a padded torch image tensor.
    Supports:
        (H,W)
        (C,H,W)
    """

    img = img_padded.detach().cpu()

    # --------------------------------------------------------
    # CHW -> HWC
    # --------------------------------------------------------

    if img.ndim == 3:

        img = img.permute(1, 2, 0)

    img = img.numpy()

    plt.figure(figsize=(5, 5))

    if img.ndim == 2:

        plt.imshow(img, cmap="gray")

    else:

        plt.imshow(img)

    plt.title(title)

    plt.axis("off")

    plt.tight_layout()

    plt.savefig("test.png")

    plt.close()

from PIL import Image
import numpy as np


def save_padded_image(
    img_padded,
    save_path="test.png"
):
    """
    Saves a padded torch image tensor.

    Supports:
        (H,W)
        (C,H,W)
    """

    img = img_padded.detach().cpu()

    # --------------------------------------------------------
    # CHW -> HWC
    # --------------------------------------------------------

    if img.ndim == 3:

        img = img.permute(1, 2, 0)

    img = img.numpy()

    # --------------------------------------------------------
    # FLOAT -> UINT8
    # --------------------------------------------------------

    img = (
        np.clip(img, 0, 1) * 255
    ).astype(np.uint8)

    Image.fromarray(img).save(save_path)

def preprocess_image_noresize(image_path, max_size,device="cuda:0",tag=''):
    """
    Loads a square image, pads it with zeros equally on all sides
    to reach (max_size, max_size), then flattens and returns it
    as a torch tensor.
    """

    img = (read_image(image_path).float()/255.0).to(device)  # (C, H, W)

    C, H, W = img.shape

    assert H == W, f"Image is not square: {H}x{W}"
    assert H <= max_size, f"Image size {H} exceeds max_size {max_size}"

    pad_total = max_size - H
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before

    # F.pad format for 2D images: (left, right, top, bottom)
    padding = (pad_before, pad_after, pad_before, pad_after)

    img_padded = F.pad(img, padding, mode="constant", value=0)

    if tag != '':
        save_padded_image(
            img_padded,
            f"original_images/original_{str(tag)}.png"
        )

    img_flat = img_padded.flatten()

    return img_flat

def trim_black_borders(img, tol=1e-3):
    """
    Trims black borders from a torch image tensor.

    Parameters:
    - img: torch.Tensor of shape (C, H, W)
    - tol: tolerance for considering a pixel as non-black

    Returns:
    - img_trimmed: torch.Tensor of shape (C, H_trimmed, W_trimmed)
    """

    if img.ndim != 3:
        raise ValueError("Input image must be a 3D tensor (C, H, W)")

    # Create a mask of non-black pixels
    non_black_mask = torch.any(img > tol, dim=0)

    if not non_black_mask.any():
        # If the entire image is black, return the original image
        return img

    # Find the bounding box of non-black pixels
    rows = torch.where(non_black_mask.any(dim=1))[0]
    cols = torch.where(non_black_mask.any(dim=0))[0]

    top = int(rows[0].item())
    bottom = int(rows[-1].item())
    left = int(cols[0].item())
    right = int(cols[-1].item())

    # Crop the image to the bounding box
    img_trimmed = img[:, top:bottom + 1, left:right + 1]

    return img_trimmed

def preprocess_image_resize(image_path, resize_dim,device="cuda:0",tag=''):
    """
    Loads an image, crops out black borders as much as possible while keeping the
    content centered, converts it to a square crop, resizes it to (resize_dim, resize_dim),
    and returns it as a flattened torch tensor.
    """

    img = (read_image(image_path).float()/255.0).to(device)  # (C, H, W)

    if img.ndim == 2:
        img = img.unsqueeze(0)

    C, H, W = img.shape

    # Remove black borders by finding the tight bounding box around non-black pixels.
    img_cropped = trim_black_borders(img)

    img_resized = F.interpolate(
        img_cropped.unsqueeze(0),
        size=(resize_dim, resize_dim),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    if tag != '':
        save_padded_image(
            img_resized,
            f"original_images/original_{str(tag)}.png"
        )

    img_flat = img_resized.flatten()

    return img_flat

def preprocess_image_relevant_crops(image_path,crop_size,device="cuda:0",tag=''):
    """
    Loads an image, trims black borders first, finds three points on the horizontal midline
    of the trimmed image at the center of the left, center, and right thirds of the image,
    crops a square region of size (crop_size, crop_size) around each point, and returns a
    dictionary of flattened torch tensors for each crop. If the image is a left eye image the left crop is assigned to the "OD" key, else the right crop is assigned to the "OD" key, the center crop is assigned to "macula", and the
    remaining crop is assigned to "other".
    """

    img = (read_image(image_path).float()/255.0).to(device)  # (C, H, W)

    if img.ndim == 2:
        img = img.unsqueeze(0)

    img_trimmed = trim_black_borders(img)

    _, height, width = img_trimmed.shape
    mid_y = height // 2

    x_positions = [int(width / 6), width // 2, int(5 * width / 6)]

    def extract_square_crop(image, center_y, center_x, size):
        channels, img_h, img_w = image.shape

        start_y = center_y - size // 2
        end_y = start_y + size
        start_x = center_x - size // 2
        end_x = start_x + size

        if start_y < 0:
            overflow = -start_y
            start_y = 0
            end_y = min(img_h, end_y + overflow)
        elif end_y > img_h:
            overflow = end_y - img_h
            end_y = img_h
            start_y = max(0, start_y - overflow)

        if start_x < 0:
            overflow = -start_x
            start_x = 0
            end_x = min(img_w, end_x + overflow)
        elif end_x > img_w:
            overflow = end_x - img_w
            end_x = img_w
            start_x = max(0, start_x - overflow)

        crop = torch.zeros((channels, size, size), dtype=image.dtype, device=image.device)

        src_y0 = max(0, start_y)
        src_y1 = min(img_h, end_y)
        src_x0 = max(0, start_x)
        src_x1 = min(img_w, end_x)

        dst_y0 = max(0, -start_y)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = max(0, -start_x)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        if src_y1 > src_y0 and src_x1 > src_x0:
            crop[:, dst_y0:dst_y1, dst_x0:dst_x1] = image[:, src_y0:src_y1, src_x0:src_x1]
        assert crop.shape == (channels, size, size), f"Crop shape mismatch: {crop.shape} != {(channels, size, size)}"
        return crop

    left_crop = extract_square_crop(img_trimmed, mid_y, x_positions[0], crop_size)
    center_crop = extract_square_crop(img_trimmed, mid_y, x_positions[1], crop_size)
    right_crop = extract_square_crop(img_trimmed, mid_y, x_positions[2], crop_size)

    filename = image_path.split('/')[-1]
    df = pd.read_csv("LvsR.csv")
    row = df[df['FileName'] == filename]
    label = row['Label'].values[0] if not row.empty else None

    if label == 'L':
        od_crop = left_crop
        other_crop = right_crop
    elif label == 'R':
        od_crop = right_crop
        other_crop = left_crop
    else:
        raise ValueError(f"Label for {filename} not found in LvsR.csv")

    if tag != '':
        save_padded_image(
            od_crop,
            f"original_images/{str(tag)}_od.png"
        )
        save_padded_image(
            center_crop,
            f"original_images/{str(tag)}_center.png"
        )
        save_padded_image(
            other_crop,
            f"original_images/{str(tag)}_other.png"
        )

    return {
        "OD": od_crop.flatten(),
        "macula": center_crop.flatten(),
        "other": other_crop.flatten(),
    }

def center_matrix_batchwise_torch(X, num_batches=10):
    """
    Row-centers a large matrix X (d x n) in batches using PyTorch.

    Parameters
    ----------
    X : torch.Tensor
        Data matrix of shape (d, n), where columns are samples.
    num_batches : int
        Number of row-batches to split into.

    Returns
    -------
    X_centered : torch.Tensor
        Row-centered matrix of same shape as X.
    mean_image : torch.Tensor
        Mean image of shape (d, 1).
    """

    d, n = X.shape
    batch_size = d // num_batches
    remainder = d % num_batches

    centered_batches = []
    mean_batches = []

    start = 0

    for i in tqdm(range(num_batches)):

        extra = 1 if i < remainder else 0
        end = start + batch_size + extra

        X_batch = X[start:end, :]   # (batch_rows, n)

        mean_batch = X_batch.mean(dim=1, keepdim=True,)  # (batch_rows, 1)
        X_batch_centered = X_batch - mean_batch

        centered_batches.append(X_batch_centered)
        mean_batches.append(mean_batch)

        start = end

    X_centered = torch.cat(centered_batches, dim=0)
    mean_image = torch.cat(mean_batches, dim=0)

    return X_centered, mean_image

def standardize_rows_batchwise_torch(X_centered, num_batches=10, eps=1e-8):
    """
    Divides a row-centered matrix by its row-wise standard deviation (batchwise).

    Parameters
    ----------
    X_centered : torch.Tensor
        Row-centered matrix of shape (d, n).
    num_batches : int
        Number of row batches.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    X_standardized : torch.Tensor
        Row-wise standardized matrix.
    row_std : torch.Tensor
        Row-wise standard deviation of shape (d, 1).
    """

    d, n = X_centered.shape
    batch_size = d // num_batches
    remainder = d % num_batches

    standardized_batches = []
    std_batches = []

    start = 0

    for i in tqdm(range(num_batches)):
        extra = 1 if i < remainder else 0
        end = start + batch_size + extra

        X_batch = X_centered[start:end, :]  # (batch_rows, n)

        # Since already centered
        var_batch = torch.sum(X_batch**2, dim=1, keepdim=True) / (n - 1)
        std_batch = torch.sqrt(var_batch) + eps

        X_batch_std = X_batch / std_batch

        standardized_batches.append(X_batch_std)
        std_batches.append(std_batch)

        start = end

    X_standardized = torch.cat(standardized_batches, dim=0)
    row_std = torch.cat(std_batches, dim=0)

    return X_standardized, row_std

import csv

def save_list_to_csv(data, filename):
    """
    Saves a list to a CSV file.

    Parameters:
    data : list
        List of values or list of lists.
    filename : str
        Output CSV file path.
    """
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        
        # If it's a list of lists → write rows
        if isinstance(data[0], (list, tuple)):
            writer.writerows(data)
        else:
            # Single list → write one column
            for item in data:
                writer.writerow([item])

def analyze_precomputed_eigenvalues(eigenvalues, beta=0.01, title_suffix=""):
    """
    Performs scree analysis on precomputed eigenvalues.

    Parameters:
    - eigenvalues: 1D array of eigenvalues (need not be sorted)
    - beta: acceptable variance loss (0 to 1).
            Keeps components capturing at least (1 - beta) variance.
    - title_suffix: optional string for plot titles
    """
    
    variance_cutoff = 1 - beta
    
    # 1. Sort eigenvalues (descending)
    eigenvalues = np.array(eigenvalues.cpu())
    eigenvalues = np.sort(eigenvalues)[::-1]
    save_list_to_csv(eigenvalues,f"eigenvalues_{title_suffix}.csv")
    
    # Remove tiny negative values due to numerical noise
    eigenvalues[eigenvalues < 0] = 0
    
    # 2. Variance metrics
    total_variance = np.sum(eigenvalues)
    
    if total_variance == 0:
        raise ValueError("Total variance is zero. Check eigenvalues.")
    
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # 3. Select k
    k_components = np.searchsorted(cumulative_variance, variance_cutoff) + 1

    fro_norm_error = np.sum(eigenvalues[k_components:])
    
    print(f"--- Eigenvalue Analysis {title_suffix} ---")
    print(f"Total Eigenvalues: {len(eigenvalues)}")
    print(f"Beta (acceptable variance loss): {beta}")
    print(f"Variance cutoff (1 - beta): {variance_cutoff:.2%}")
    print(f"Components to keep: {k_components}")
    print(f"Actual Variance Explained: {cumulative_variance[k_components-1]:.2%}")
    print(f"Frobenius Norm Error: {fro_norm_error}")
    print("-" * 40)

    # 4. Plot
    components = np.arange(1, len(eigenvalues) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Variance explained
    plt.subplot(1, 2, 1)
    plt.bar(components, explained_variance_ratio, alpha=0.5, label='Individual Variance')
    plt.step(components, cumulative_variance, where='mid',
             label='Cumulative Variance', color='red')
    plt.axhline(y=variance_cutoff, color='green', linestyle='--',
                label=f'Cutoff (1-β={variance_cutoff:.2%})')
    plt.axvline(x=k_components, color='green', linestyle='--')
    plt.title(f'Variance Explained {title_suffix}')
    plt.xlabel('Component Index')
    plt.ylabel('Explained Variance Ratio')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # Plot 2: Classical Scree
    plt.subplot(1, 2, 2)
    plt.plot(components, eigenvalues, 'o-', linewidth=2)
    plt.axvline(x=k_components, color='green', linestyle='--',
                label=f'Cutoff (k={k_components})')
    plt.yscale('log')
    plt.title(f'Classical Scree Plot {title_suffix} (Log Scale)')
    plt.xlabel('Component Index')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{title_suffix}.png")
    
    return k_components, eigenvalues[:k_components], cumulative_variance[k_components-1], fro_norm_error

def beta_vs_fro_norm(eigenvalues, beta_lst, title_suffix=""):
    fro_norm_sq_list = []
    k_lst = []
    explained_variance_cutoff = []

    for beta in beta_lst:
        k, _, _, fro_norm_sq = analyze_precomputed_eigenvalues(
            eigenvalues,
            beta,
            title_suffix + f"_beta_{beta}"
        )

        fro_norm_sq_list.append(fro_norm_sq)
        k_lst.append(k)

        # X-axis for first subplot
        explained_variance_cutoff.append(1 - beta)

    # Convert to numpy arrays for cleaner handling
    fro_norm_sq_list = np.array(fro_norm_sq_list)
    k_lst = np.array(k_lst)
    explained_variance_cutoff = np.array(explained_variance_cutoff)

    # ============================================================
    # Plotting
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ------------------------------------------------------------
    # Subplot 1:
    # Frobenius norm squared vs explained variance cutoff
    # ------------------------------------------------------------
    axes[0].plot(
        explained_variance_cutoff,
        fro_norm_sq_list,
        marker='o'
    )

    axes[0].set_xlabel("Explained Variance Cutoff (1 - beta)")
    axes[0].set_ylabel("Frobenius Norm Squared")
    axes[0].set_title("Fro Norm vs Explained Variance Cutoff")
    axes[0].grid(True)

    # ------------------------------------------------------------
    # Subplot 2:
    # Frobenius norm squared vs k
    # ------------------------------------------------------------
    axes[1].plot(
        k_lst,
        fro_norm_sq_list,
        marker='o'
    )

    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Frobenius Norm Squared")
    axes[1].set_title("Fro Norm vs k")
    axes[1].grid(True)

    # Overall title
    if title_suffix != "":
        fig.suptitle(title_suffix)

    plt.tight_layout()
    plt.savefig(f"FrobeniusNormSquared_{title_suffix}.png")

    return {
        "beta_lst": beta_lst,
        "explained_variance_cutoff": explained_variance_cutoff,
        "k_lst": k_lst,
        "fro_norm_sq_list": fro_norm_sq_list,
    }


def compute_projection_Z(eigenvalues, eigenvectors, k):
    """
    Computes the reduced representation Z from eigenvalues and eigenvectors
    of H^T H.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues (length n).
    eigenvectors : np.ndarray
        Matrix of eigenvectors (n x n), columns are eigenvectors.
    k : int
        Number of principal components to keep.

    Returns
    -------
    Z : np.ndarray
        Reduced representation (k x n).
    """

    # sort eigenvalues descending
    eigenvalues = np.array(eigenvalues.cpu())
    eigenvectors = np.array(eigenvectors.cpu())
    idx = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # take top-k
    lambda_k = eigenvalues[:k]
    V_k = eigenvectors[:, :k]

    # compute Z = sqrt(Λ_k) V_k^T
    Z = np.sqrt(lambda_k)[:, None] * V_k.T

    Z = torch.from_numpy(Z)

    return Z