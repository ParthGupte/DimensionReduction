import torch
from svd_helper import *

HTH = torch.load("saved_matrices/HTH.pt").to("cpu")

mu = torch.load("saved_matrices/mu.pt").unsqueeze(1).to("cpu")

sigma = torch.load("saved_matrices/sigma.pt").to("cpu")

sigma_inv = 1/sigma

DEVICE = mu.device

print(HTH.shape,mu.shape)
n = int(HTH.shape[0])
d = int(mu.shape[0])
print(n,d)
one = torch.ones((n,1),device=DEVICE)
HTH_cov = (HTH + (one @ mu.T @ mu @ one.T) - (one @ one.T @ HTH/n) - (HTH/n @ one @ one.T))/n

HTH_cor = sigma_inv[:,None] * HTH_cov * sigma_inv[None,:] 

eigenvalues_raw, eigenvectors_raw = torch.linalg.eigh(HTH)

analyze_precomputed_eigenvalues(eigenvalues_raw,title_suffix="RawMoments")

eigenvalues_cov, eigenvectors_cov = torch.linalg.eigh(HTH_cov)

analyze_precomputed_eigenvalues(eigenvalues_cov,title_suffix="Covariance")

eigenvalues_cor, eigenvectors_cor = torch.linalg.eigh(HTH_cor)

analyze_precomputed_eigenvalues(eigenvalues_cor,title_suffix="Correlation")

