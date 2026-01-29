import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF

def rmsd(A, B):

    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    A_rot = A @ R
    return np.sqrt(((A_rot - B)**2).sum() / A.shape[0])

def gpr_kernel(coord):
    nframe = coord.shape[0]
    D = np.zeros((nframe, nframe))
    for i in range(nframe):
        for j in range(i, nframe):
            d = rmsd(coord[i], coord[j])
            D[i, j] = D[j, i] = d
    length_scale = np.median(D)
    rbf = RBF(length_scale=length_scale)
    K = rbf(D)
    diag = np.sqrt(np.diag(K))
    K_norm = K / np.outer(diag, diag)
    return K_norm
