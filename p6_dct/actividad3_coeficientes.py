import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from p4_color.conversion import rgb_to_ycrcb
from p6_dct.dct import mi_dct, mi_idct
from p5_muestreo.muestreo import rmse

rgb = np.array(Image.open('p4_color/datos/escena_referencia.png').convert('RGB'))
Yq, _, _ = rgb_to_ycrcb(rgb)
img = Yq.astype(np.float64)
x = img[704:712, 432:440]  # Select the rows and columns of the marked block.
X = mi_dct(x)
# These settings affect only printing, without rounding the stored matrix values.
np.set_printoptions(precision=2, suppress=True, linewidth=120)
print('DETAILED BLOCK - signed DCT coefficients:')
print(X)

# Sort by absolute value and select the four largest positions.
idx = np.argsort(np.abs(X), axis=None)[::-1][:4]
# Obtain the row and column of each selected position.
filas, columnas = np.unravel_index(idx, X.shape)
# Iterate over row and column indices together.
print('\nDETAILED BLOCK - four largest coefficients by absolute value:')
for f, c in zip(filas, columnas):
    print(f'X[{f},{c}] = {X[f,c]:9.3f}   position counting from 1: ({f+1},{c+1})')

    # Reconstruct the nearly uniform block using only DC.
plano = img[0:8, 0:8]
Xp = mi_dct(plano)   # Calculate the DCT of plano using mi_dct.
solo_dc = np.zeros_like(Xp)
solo_dc[0, 0] = Xp[0, 0]   # Copy the DC coefficient from Xp.
print('\nNEARLY UNIFORM BLOCK - top-left corner of the path image')
print(f'Reconstruction using only DC: RMSE={rmse(plano, mi_idct(solo_dc)):.3f} Yq levels')

# Enter the six table positions using Python indices.
posiciones = [
    (0, 0),  # First coefficient.
    (0, 1),  # Second coefficient.
    (1, 0),  # Third coefficient.
    (3, 0),  # Fourth coefficient.
    (6, 5),  # Fifth coefficient.
    (1, 7),  # Sixth coefficient.
]

# Repeat with 1, 2, ..., 6 coefficients, adding them in the listed order.
print('\nDETAILED BLOCK - block marked in the image')
print('Reconstructions using the table positions. Errors in Yq levels:')
for k in range(1, len(posiciones) + 1):
    # Start each test with the same shape and type as X, but all values set to zero.
    Xk = np.zeros_like(X)
    # posiciones[:k] selects the first k positions in the list.
    for f, c in posiciones[:k]:
        Xk[f, c] = X[f, c]   # retain the coefficient from X
    # The IDCT reconstructs the block using only the copied coefficients.
    xk = mi_idct(Xk)   # Reconstruct the block from Xk using mi_idct.
    print(f'{k} coeff.  maxdif={np.max(np.abs(x - xk)):7.3f}  RMSE={rmse(x, xk):7.3f}')

# Retain the four coefficients with the largest absolute values.
X4 = np.zeros_like(X)
# filas and columnas contain the four pairs from argsort and unravel_index.
X4[filas, columnas] = X[filas, columnas]   # retain only the four selected coefficients
x4 = mi_idct(X4)   # Reconstruct the block from X4 using mi_idct.
print('\nDETAILED BLOCK - four largest coefficients by absolute value')
print(f'4 largest  maxdif={np.max(np.abs(x - x4)):7.3f}  RMSE={rmse(x, x4):7.3f}')

fig, ax = plt.subplots(2, 4, figsize=(12, 6))
# ravel allows all eight panels to be indexed from ax[0] to ax[7].
ax = ax.ravel()
ax[0].imshow(x, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
ax[0].set_title('Original (64 coeff.)')
for i, n in enumerate(range(1, 7)):
    Xk = np.zeros_like(X)
    for f, c in posiciones[:n]:
        Xk[f, c] = X[f, c]
    ax[i+1].imshow(mi_idct(Xk), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax[i+1].set_title(f'{n} coefficient(s)')
ax[7].imshow(x4, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
ax[7].set_title('4 largest')
for a in ax: a.axis('off')
plt.tight_layout()
plt.savefig('p6_dct/resultados/actividad3_coeficientes.png', dpi=100, bbox_inches='tight')
plt.show()