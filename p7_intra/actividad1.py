import numpy as np
from p6_dct.dct import matriz_dct, mi_dct, mi_idct
from p7_intra.apoyo import cargar_y, dct_imagen, seleccion, figura_dct, figura_bloques

imagen = cargar_y('p4_color/datos/escena_referencia.png')
T = matriz_dct(8)   # Built once; the same matrix was checked in P6.
D_sin = dct_imagen(imagen, T)
D_centrada = dct_imagen(imagen - 128, T)   # Apply the level shift before the transform.
print(f'Observed Yq: {imagen.min():.0f} to {imagen.max():.0f}')
print(f'Unshifted: minimum {D_sin.min():.4f}; maximum {D_sin.max():.4f}')
print(f'Shifted:   minimum {D_centrada.min():.4f}; maximum {D_centrada.max():.4f}')
figura_dct(imagen, D_sin, D_centrada)

# Extract the four selected 8x8 blocks from the reference image.
bloques = seleccion(imagen)
coeficientes = {}    # Store each block DCT for the final figure.
for nombre, bloque in bloques.items():
    X = mi_dct(bloque - 128)    # Apply the level shift to the block before the DCT.
    coeficientes[nombre] = X
    rec = mi_idct(X) + 128    # Apply the IDCT to X and restore the original levels.

    # Compare the coefficients before and after level shifting.
    diferencia = X - mi_dct(bloque)
    ac = diferencia.copy()
    ac[0, 0] = 0    # Exclude DC to measure the change only at AC positions.

    # Print the DC change, largest AC change and maximum pixel error.
    print(f'{nombre:18s} DC_change={diferencia[0,0]:.4f} '
          f'max_AC_change={np.abs(ac).max():.3e} '
          f'reconstruction_error={np.abs(bloque-rec).max():.3e}')
figura_bloques(imagen, bloques, coeficientes)