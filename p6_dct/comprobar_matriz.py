from p6_dct.dct import matriz_dct
import numpy as np

T = matriz_dct(8)
# np.eye(8) creates the 8x8 identity: ones on the diagonal and zeros elsewhere.
# If T is orthogonal, T @ T.T equals this identity matrix.
print(np.max(np.abs(T @ T.T - np.eye(8))))   # approximately 1e-15