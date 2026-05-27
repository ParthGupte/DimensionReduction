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

R = torch.randn((n,MAX_DIM))

vec_i_batch = []
for i, f_i in tqdm(enumerate(image_files)):
    vec_i = preprocess_image_noresize(f_i,MAX_SIZE,DEVICE)
    vec_i_batch.append(vec_i)
    if len(vec_i_batch) == CALC_BATCH_SIZE:
        vec_i_batch_torch = torch.stack(vec_i_batch)


        print(vec_i_batch_torch.shape,vec_i_batch_torch.device)
        break