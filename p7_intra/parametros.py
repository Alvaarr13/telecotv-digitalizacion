"""Parámetros originales de la P7 de Televisión.
Pesos y recorrido extraídos de variables_cod_intra.mat del material docente.
ORDEN usa índices 0-based sobre una matriz NumPy aplanada por filas.
La simplificación didáctica de la cuantificación no implementa MPEG-2 completo.
"""
import numpy as np

PESOS = np.array([[8, 16, 19, 22, 26, 27, 29, 34], [16, 16, 22, 24, 27, 29, 34, 37], [19, 22, 26, 27, 29, 34, 34, 38], [22, 22, 26, 27, 29, 34, 37, 40], [22, 26, 27, 29, 32, 35, 40, 48], [26, 27, 29, 32, 35, 40, 48, 58], [26, 27, 29, 34, 38, 46, 56, 69], [27, 29, 35, 38, 46, 56, 69, 83]], dtype=np.float64)

ORDEN = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63], dtype=int)

FACTORES = (1, 4, 10, 30)
BLOQUES = {"uniforme": (0, 0), "borde vertical": (704, 432), "vegetacion": (456, 816), "borde horizontal": (752, 416)}
