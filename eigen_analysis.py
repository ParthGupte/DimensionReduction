import torch
from svd_helper import *

HTH = torch.load("saved_matrices/HTH.pt").to("cpu")

mu = torch.load("saved_matrices/mu.pt").unsqueeze(1).to("cpu")

sigma = torch.load("saved_matrices/sigma.pt").to("cpu")

sigma_inv = 1/sigma

DEVICE = mu.device

BETA_LST = [0.01,0.05,0.1,0.15,0.2]

print(HTH.shape,mu.shape)
n = int(HTH.shape[0])
d = int(mu.shape[0])
print(n,d)
one = torch.ones((n,1),device=DEVICE)
HTH_cov = (HTH + (one @ mu.T @ mu @ one.T) - (one @ one.T @ HTH/n) - (HTH/n @ one @ one.T))/n

HTH_cor = sigma_inv[:,None] * HTH_cov * sigma_inv[None,:] 

eigenvalues_raw, eigenvectors_raw = torch.linalg.eigh(HTH)

torch.save(eigenvectors_raw,"saved_matrices/eigenvectors_raw.pt")

k_raw, _, _, _ = analyze_precomputed_eigenvalues(eigenvalues_raw,title_suffix="RawMoments")

beta_vs_fro_norm(eigenvalues_raw,BETA_LST,title_suffix="RawMoments")

Z_raw = compute_projection_Z(eigenvalues_raw,eigenvectors_raw,k_raw)

torch.save(Z_raw,"saved_matrices/Z_raw.pt")

eigenvalues_cov, eigenvectors_cov = torch.linalg.eigh(HTH_cov)

torch.save(eigenvectors_cov,"saved_matrices/eigenvectors_cov.pt")

k_cov, _, _,_ = analyze_precomputed_eigenvalues(eigenvalues_cov,title_suffix="Covariance")

beta_vs_fro_norm(eigenvalues_cov,BETA_LST,title_suffix="Covariance")

Z_cov = compute_projection_Z(eigenvalues_cov,eigenvectors_cov,k_cov)

torch.save(Z_cov,"saved_matrices/Z_cov.pt")

eigenvalues_cor, eigenvectors_cor = torch.linalg.eigh(HTH_cor)

torch.save(eigenvectors_cor,"saved_matrices/eigenvectors_cor.pt")

k_cor,_,_,_ = analyze_precomputed_eigenvalues(eigenvalues_cor,title_suffix="Correlation")

beta_vs_fro_norm(eigenvalues_cor,BETA_LST,title_suffix="Correlation")

Z_cor = compute_projection_Z(eigenvalues_cor,eigenvectors_cor,k_cor)

torch.save(Z_cor,"saved_matrices/Z_cor.pt")
