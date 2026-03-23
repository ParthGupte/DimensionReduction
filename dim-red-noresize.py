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
IMAGE_DIR = "MESSIDOR/images"

DEVICE = "cuda:0"

CALC_BATCH_SIZE = 100

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

HTH = torch.zeros((n,n),device=DEVICE)
avg_img = torch.zeros((MAX_SIZE**2*3,),device=DEVICE)
vec_i_batch = []
for i, f_i in tqdm(enumerate(image_files)):
    vec_i = preprocess_image_noresize(f_i,MAX_SIZE,DEVICE)
    avg_img += vec_i
    vec_i_batch.append(vec_i)
    if len(vec_i_batch) == CALC_BATCH_SIZE:
        vec_i_batch_torch = torch.stack(vec_i_batch)
        # print(vec_i_batch_torch.shape,vec_i_batch_torch.device)
        if vec_i is not None:
            vec_j_batch = []
            for j, f_j in tqdm(enumerate(image_files[:i+1])):
                vec_j = preprocess_image_noresize(f_j,MAX_SIZE,DEVICE)
                vec_j_batch.append(vec_j)
                if len(vec_j_batch) == CALC_BATCH_SIZE:
                    vec_j_batch_torch = torch.stack(vec_j_batch)
                    # print(vec_j_batch_torch.shape,vec_j_batch_torch.device)
                    if vec_j is not None:
                        val = torch.matmul(vec_i_batch_torch,vec_j_batch_torch.T)
                        HTH[i-CALC_BATCH_SIZE+1:i+1,j-CALC_BATCH_SIZE+1:j+1] = val
                        HTH[j-CALC_BATCH_SIZE+1:j+1,i-CALC_BATCH_SIZE+1:i+1] = val.T
                    vec_j_batch = []
            else:
                size_j = len(vec_j_batch)
                if size_j > 0:
                    vec_j_batch_torch = torch.stack(vec_j_batch)
                    # print(vec_j_batch_torch.shape,vec_j_batch_torch.device)
                    val = torch.matmul(vec_i_batch_torch,vec_j_batch_torch.T)
                    HTH[i-CALC_BATCH_SIZE+1:i+1,j-size_j+1:j+1] = val
                    HTH[j-size_j+1:j+1,i-CALC_BATCH_SIZE+1:i+1] = val.T
        vec_i_batch = []
else:
    size_i = len(vec_i_batch)
    if size_i > 0:
        vec_i_batch_torch = torch.stack(vec_i_batch)
        # print(vec_i_batch_torch.shape,vec_i_batch_torch.device)
        vec_j_batch = []
        for j, f_j in tqdm(enumerate(image_files[:i+1])):
            vec_j = preprocess_image_noresize(f_j,MAX_SIZE,DEVICE)
            vec_j_batch.append(vec_j)
            if len(vec_j_batch) == CALC_BATCH_SIZE:
                vec_j_batch_torch = torch.stack(vec_j_batch)
                # print(vec_j_batch_torch.shape,vec_j_batch_torch.device)
                if vec_j is not None:
                    val = torch.matmul(vec_i_batch_torch,vec_j_batch_torch.T)
                    HTH[i-size_i+1:i+1,j-CALC_BATCH_SIZE+1:j+1] = val
                    HTH[j-CALC_BATCH_SIZE+1:j+1,i-size_i+1:i+1] = val.T
                vec_j_batch = []
        else:
            size_j = len(vec_j_batch)
            if size_j > 0:
                vec_j_batch_torch = torch.stack(vec_j_batch)
                # print(vec_j_batch_torch.shape,vec_j_batch_torch.device)
                val = torch.matmul(vec_i_batch_torch,vec_j_batch_torch.T)
                HTH[i-size_i+1:i+1,j-size_j+1:j+1] = val
                HTH[j-size_j+1:j+1,i-size_i+1:i+1] = val.T
avg_img /= n  

sigma = torch.zeros((n))

for i, f_i in tqdm(enumerate(image_files)):
    vec_i = preprocess_image_noresize(f_i,MAX_SIZE,DEVICE)
    sigma[i] = torch.sqrt(torch.sum((vec_i-avg_img)**2))


# Construct Data Matrix H
# Rows = flattened features, Columns = images
# H = torch.column_stack(data_vectors)

print(f"Data Matrix H shape: {HTH.shape}",HTH.max(),HTH.min())

torch.save(HTH,"saved_matrices/HTH.pt")
torch.save(avg_img,"saved_matrices/mu.pt")
torch.save(sigma,"saved_matrices/sigma.pt")


# H_raw = H
# H_cov, _ = center_matrix_batchwise_torch(H_raw,50)
# H_cor, _ = standardize_rows_batchwise_torch(H_cov,1000)

