import numpy as np
from p6_dct.dct import matriz_dct
from p7_intra.parametros import PESOS, ORDEN
from p7_intra.apoyo import extender, iter_bloques

T = matriz_dct(8)

def redondear(x):
    # Values halfway between two integers are rounded away from zero.
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def validar(X, FC, pesos):
    if not isinstance(X, np.ndarray) or X.shape != (8, 8) or X.dtype != np.float64:
        raise ValueError('An 8x8 float64 block is required')
    if not np.isfinite(X).all():
        raise ValueError('The block must contain finite values')
    if not isinstance(FC, (int, np.integer)) or isinstance(FC, (bool, np.bool_)) or not 1 <= FC <= 112:
        raise ValueError('FC must be an integer from 1 to 112')
    if pesos.shape != (8, 8) or not np.isfinite(pesos).all() or np.any(pesos <= 0):
        raise ValueError('Weights must be positive and finite, with shape 8x8')

def cuantificar(X, FC, pesos=PESOS):
    validar(X, FC, pesos)
    indices = redondear(X / (FC * pesos))   # Apply the AC quantization expression.
    indices[0, 0] = X[0, 0] / FC           # Retain DC without quantization.
    return indices

def zigzag(indices):
    # ORDEN gives the original scan order using NumPy indices.
    return indices.ravel()[ORDEN]

def inversa_zigzag(vector):
    if np.asarray(vector).shape != (64,):
        raise ValueError('The vector must have 64 elements')
    matriz = np.empty(64, dtype=np.float64)
    matriz[ORDEN] = vector
    return matriz.reshape(8, 8)

def ceros_finales(vector):
    if np.asarray(vector).shape != (64,):
        raise ValueError('The vector must have 64 elements')
    # Obtain positions of non-zero values, rather than the values themselves.
    no_nulos = np.flatnonzero(vector != 0)
    if no_nulos.size == 0:
        return vector.size    # Return the number of elements: all of them are zero.
    ultima_posicion = no_nulos[-1]
    return vector.size - ultima_posicion - 1        # Count the positions remaining to the end of the vector.

def cuantificacion_inversa(indices, FC, pesos=PESOS):
    validar(indices, FC, pesos)
    Xr = redondear(indices * FC * pesos)    # Reverse the AC scaling using the same weights and FC.
    Xr[0, 0] = indices[0, 0] * FC         # Restore the retained DC value without rounding it.
    return Xr

def cod_intra(imagen, FC, pesos=PESOS, desplazamiento=128):
    extendida, forma = extender(imagen)
    indices = np.empty_like(extendida)
    for fila, columna, bloque in iter_bloques(extendida):
        # T is the P6 matrix, calculated once when this module is imported.
        X = T @ (bloque - desplazamiento) @ T.T   # Apply the matrix DCT to (bloque - desplazamiento).
        # Quantize X with FC and pesos; store the result at this block position.
        indices[fila:fila+8, columna:columna+8] = cuantificar(X, FC, pesos)
    return indices, forma

def decod_intra(indices, forma, FC, pesos=PESOS, desplazamiento=128):
    recuperada = np.empty_like(indices)
    for fila, columna, bloque in iter_bloques(indices):
        Xr = cuantificacion_inversa(bloque, FC, pesos)
        recuperada[fila:fila+8, columna:columna+8] = T.T @ Xr @ T    # Apply the matrix IDCT to Xr using T.T and T.
    h, w = forma
    return recuperada[:h, :w] + desplazamiento