import numpy as np
from scipy.fft import dctn, idctn

def matriz_dct(N=8):
    # Row u = frequency pattern u; column v = spatial position.
    # Allocate an NxN matrix; the loop calculates each entry.
    T = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            # The first row represents DC and has the same weight at every position.
            if u == 0:
                T[u, v] = np.sqrt(1/N)              # weight w(0)
            else:
                # Python indices start at 0: (2v-1)(u-1) becomes (2v+1)u.
                T[u, v] = np.sqrt(2/N) * np.cos(np.pi * ((2*v + 1) * u / (2*N)))
    return T

def mi_dct(x):
    # isinstance checks whether x is a NumPy array.
    # dtype gives the data type; float64 is required here.
    if not isinstance(x, np.ndarray) or x.dtype != np.float64:
        # raise stops the call and displays the reason for the error.
        raise TypeError('mi_dct: input must be a float64 array')

    # shape contains (rows, columns): only 8x8 blocks are accepted.
    if x.shape != (8, 8):
        raise ValueError('mi_dct: input must be an 8x8 block')

    # Build the DCT matrix for this block size.
    T = matriz_dct(8)
    # Complete the product that transforms pixels into coefficients.
    return T @ x @ T.T   # matrix product for the 2D DCT

def mi_idct(X):
    # X must contain the coefficients in a NumPy float64 array.
    # If the type is incorrect, raise stops the call with a message.
    if not isinstance(X, np.ndarray) or X.dtype != np.float64:
        raise TypeError('mi_idct: input must be a float64 array')

    # Check for 8 rows and 8 columns, as in the forward DCT.
    if X.shape != (8, 8):
        raise ValueError('mi_idct: input must be an 8x8 block')

    T = matriz_dct(8)
    # Complete the inverse product to recover the block values.
    return T.T @ X @ T   # matrix product for the inverse transform