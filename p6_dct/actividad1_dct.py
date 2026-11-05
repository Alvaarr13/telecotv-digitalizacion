import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import dctn
from p4_color.conversion import rgb_to_ycrcb
from p6_dct.dct import mi_dct

# Open the images as RGB and obtain luminance using the P4 function.
def cargar_y(ruta):
    rgb = np.array(Image.open(ruta).convert('RGB'))
    # Keep Yq; the two underscores receive Crq and Cbq, which are not used here.
    Yq, _, _ = rgb_to_ycrcb(rgb)
    # Convert to float64 before calling mi_dct, keeping the same Yq levels.
    return Yq.astype(np.float64)

img = cargar_y('p4_color/datos/escena_referencia.png')
cap = cargar_y('p4_color/datos/escena_capturada.png')
h, w = cap.shape
# Each crop has 8 rows and 8 columns. The capture uses its own coordinates.
bloques = {
    'uniforme': img[0:8, 0:8],
    'detalle': img[704:712, 432:440],
    'captura': cap[h//2:h//2+8, w//2:w//2+8],
}

etiquetas = {'uniforme': 'uniform', 'detalle': 'detail', 'captura': 'capture'}

# Display the three crops with the same luminance scale.
fig_bloques, ejes = plt.subplots(1, 3, figsize=(9, 3), layout='constrained')
for eje, (nombre, x) in zip(ejes, bloques.items()):
    eje.imshow(x, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    eje.set_title(etiquetas[nombre])
    eje.axis('off')


# Calculate both DCTs for each block and keep them for the final figure.
resultados = {}
print('block        DC       error_max_DCT')
# items iterates over the name and contents of each block.
for nombre, x in bloques.items():
    X = mi_dct(x)   # Apply mi_dct to block x.
    # Use the same normalisation as in matrix T.
    X_ref = dctn(x, norm='ortho')
    error = np.max(np.abs(X - X_ref))   # Largest absolute difference between X and X_ref.
    # np.abs and np.max were already used in comprobar_matriz.py.
    # np is the short name for NumPy, imported at the start.
    # Subtract the matrices, take absolute values and find the largest.
    print(f'{etiquetas[nombre]:9s} {X[0, 0]:9.3f} {error:.3e}')
    # Store both coefficient matrices for this block together.
    resultados[nombre] = (X, X_ref)

# Each row shows a block, its own DCT and the SciPy DCT.
fig, ax = plt.subplots(3, 3, figsize=(11, 10), layout='constrained')
# enumerate adds the row index to position each block in the figure.
for i, (nombre, x) in enumerate(bloques.items()):
    X, X_ref = resultados[nombre]
    ax[i, 0].imshow(x, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax[i, 0].set_title(etiquetas[nombre] + ' - Yq')
    # Plot coefficient magnitudes using the same scale for both DCTs.
    for j, (A, titulo) in enumerate([(X, 'mi_dct'), (X_ref, 'SciPy')], 1):
        im = ax[i, j].imshow(np.abs(A), cmap='jet', vmin=0, vmax=np.abs(X_ref).max())
        ax[i, j].set_title('|' + titulo + '|')
        fig.colorbar(im, ax=ax[i, j], shrink=0.7)
for a in ax.flat:
    a.axis('off')
plt.savefig('p6_dct/resultados/actividad1_dct.png', dpi=120)

plt.show()