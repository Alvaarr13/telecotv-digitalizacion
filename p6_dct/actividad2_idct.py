import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from p4_color.conversion import rgb_to_ycrcb
from p6_dct.dct import mi_dct, mi_idct

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

# Reconstruct with all coefficients retained and check the energy.
fig, ax = plt.subplots(3, 3, figsize=(11, 10), layout='constrained')
print('block     error_max_IDCT   energia_x   energia_X')
for i, (nombre, x) in enumerate(bloques.items()):
    X = mi_dct(x)   # Calculate the DCT of block x.
    # All coefficients are used: no information has been discarded here.
    rec = mi_idct(X)   # Reconstruct the block from X using the inverse DCT.
    error = np.max(np.abs(x - rec))   # Largest absolute difference between x and rec.
    # Use np.abs and np.max, as in the comparison of the two DCTs.
    # The sum of squares measures energy; it should match before and after the DCT.
    print(f'{etiquetas[nombre]:9s} {error:.3e} {np.sum(x*x):.6f} {np.sum(X*X):.6f}')
    ax[i, 0].imshow(x, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax[i, 0].set_title(etiquetas[nombre] + ' - original')
    ax[i, 1].imshow(rec, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax[i, 1].set_title('IDCT with all coefficients')
    # The colour bar distinguishes rounding residuals from appreciable errors.
    im = ax[i, 2].imshow(np.abs(x-rec), cmap='jet', vmin=0)
    ax[i, 2].set_title('|difference| (Yq levels)')
    fig.colorbar(im, ax=ax[i, 2], shrink=0.7)
for a in ax.flat:
    a.axis('off')
plt.savefig('p6_dct/resultados/actividad2_idct.png', dpi=120)


# 4. Each function must reject both inputs.
for funcion in [mi_dct, mi_idct]:
    # The first input has an incorrect type; the second has an incorrect size.
    for prueba in [np.zeros((8, 8), dtype=np.uint8), np.zeros((4, 4))]:
        # try checks input rejection without stopping the entire script.
        try:
            funcion(prueba)
            print('FAIL:', funcion.__name__, 'did not reject the input')
        # except catches the expected error; e contains its explanatory message.
        except (TypeError, ValueError) as e:
            print('validation OK ->', e)

plt.show()