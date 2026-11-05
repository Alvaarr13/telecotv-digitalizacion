import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import dctn, idctn
from p4_color.conversion import rgb_to_ycrcb
from p5_muestreo.muestreo import rmse

etiquetas = {
    'referencia': 'reference', 'captura': 'capture',
    'camino': 'path', 'varillas': 'rods', 'captura propia': 'captured image',
    'constante': 'constant', 'todos los píxeles a cero': 'all-zero image',
}

# Open the original images without modifying their files.
rgb_referencia = np.array(Image.open('p4_color/datos/escena_referencia.png').convert('RGB'))
rgb_captura = np.array(Image.open('p4_color/datos/escena_capturada.png').convert('RGB'))
h, w = rgb_captura.shape[:2]
print(f'Original capture: {h} rows and {w} columns')
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(rgb_captura)
ax.set_title('Original capture: locate the image area without black borders')
ax.set_xlabel('Column (x)')
ax.set_ylabel('Row (y)')

# Enter the useful area limits observed in the previous figure.
fila_inicial, fila_final = 36, 460
columna_inicial, columna_final = 67, 664
# Check that the crop lies within the capture and contains pixels.
if not (0 <= fila_inicial < fila_final <= h and
        0 <= columna_inicial < columna_final <= w):
    raise ValueError('Check the useful area limits of the capture')

# Crop the capture in memory, without resizing it.
captura_util = rgb_captura[fila_inicial:fila_final, columna_inicial:columna_final]
print(f'Useful area: rows {fila_inicial}:{fila_final}, columns {columna_inicial}:{columna_final}')
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(captura_util)
ax.set_title('Cropped capture: check that the entire scene is retained')

# Keep the full reference and cropped capture for all tests.
imagenes = {}
for nombre, rgb in [('referencia', rgb_referencia), ('captura', captura_util)]:
    Yq, _, _ = rgb_to_ycrcb(rgb)
    imagenes[nombre] = Yq.astype(np.float64)
    img = imagenes[nombre]
    # Calculate the DCT of the entire area being analysed.
    X = dctn(img, norm='ortho')
    rec = idctn(X, norm='ortho')
    print(f'{etiquetas[nombre]}: Maximum reconstruction error with all coefficients retained: {np.max(np.abs(img-rec)):.3e}')

def magnitud_db(X):
    # Obtain each coefficient's magnitude using np.abs.
    Xa = np.abs(X)
    # Find the largest magnitude in Xa using .max().
    maximo = Xa.max()
    # If all values are zero, return a matrix of -inf with the same shape.
    if maximo == 0:
        return np.full(X.shape, -np.inf)
    # Keep log10(0) = -inf without displaying the NumPy warning.
    with np.errstate(divide='ignore'):
        # Express Xa in dB relative to maximo using np.log10.
        XndB = 20 * np.log10(Xa / maximo)
    return XndB

# Load the rods frame used in P5.
rgb_varillas = np.array(Image.open('p5_muestreo/datos/escena_varillas.png').convert('RGB'))
Y_varillas, _, _ = rgb_to_ycrcb(rgb_varillas)
# Collect the three images to plot them using the same procedure.
comparacion = {
    'camino': imagenes['referencia'],
    'varillas': Y_varillas.astype(np.float64),
    'captura propia': imagenes['captura'],
}

fig, ax = plt.subplots(3, 3, figsize=(12, 10), layout='constrained')
for i, (nombre, img) in enumerate(comparacion.items()):
    X = dctn(img, norm='ortho')
    XndB = magnitud_db(X)
    ax[i, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    ax[i, 0].set_title(etiquetas[nombre] + ' - Yq')
    # Show the full spectrum and its low-frequency corner.
    for j, A in [(1, XndB), (2, XndB[:64, :64])]:
        # The -120 dB limit affects only the plot; XndB retains its original values.
        im = ax[i, j].imshow(np.maximum(A, -120), cmap=plt.get_cmap('jet', 64),
                             vmin=-120, vmax=0, interpolation='nearest', aspect='auto')
        ax[i, j].set_title('Full DCT' if j == 1 else 'DCT: indices [0:64,0:64]')
        ax[i, j].set_xlabel('Horizontal frequency (v)')
        ax[i, j].set_ylabel('Vertical frequency (u)')
        fig.colorbar(im, ax=ax[i, j], label='dB relative to the maximum')
plt.savefig('p6_dct/resultados/espectros.png', dpi=120)

# Compare with the full reference or cropped capture, keeping its dimensions.
for nombre, img in imagenes.items():
    X = dctn(img, norm='ortho')
    XndB = magnitud_db(X)
    h, w = img.shape
    # Use the same crop for all four thresholds of this image.
    r, c = h//2, w//2
    # slice defines the enlarged area; min keeps it within the image boundaries.
    zona = (slice(r, min(r+128, h)), slice(c, min(c+128, w)))
    fig, ax = plt.subplots(4, 3, figsize=(12, 12), layout='constrained')
    print(f'\nImage: {etiquetas[nombre]}')
    print(f"{'Threshold dB':>12} | {'Nonzero coeff. (%)':>20} | {'RMSE (Yq levels)':>20}")
    for i, umbral in enumerate([-120, -90, -60, -30]):
        # Work on a copy: each test must start from the complete DCT.
        Xu = X.copy()   # Create a copy of X using .copy().
        # The comparison selects positions with magnitudes below the threshold.
        Xu[XndB < umbral] = 0   # Set the selected coefficients to zero.
        rec = idctn(Xu, norm='ortho')
        # Count actual nonzero values and divide by the total number of coefficients.
        pct = np.count_nonzero(Xu) / Xu.size   # Calculate the percentage using np.count_nonzero and Xu.size.
        error = rmse(img, rec)  # before clipping; over the entire image
        print(f'{umbral:12d} | {pct:20.4f} | {error:20.3f}')
        # Adding 128 displays negative differences; only the display is limited to 0-255.
        dif = np.clip(img - rec + 128, 0, 255)
        for j, (A, titulo) in enumerate([(img, 'Original'), (rec, 'Reconstruction'),
                                         (dif, 'Difference +128')]):
            ax[i, j].imshow(np.clip(A, 0, 255), cmap='gray', vmin=0, vmax=255)
            ax[i, j].set_title(f'{umbral} dB - {titulo}')
            ax[i, j].axis('off')
            # Add a small enlarged view inside each image.
            zoom = ax[i, j].inset_axes([0.53, 0.52, 0.46, 0.46])
            zoom.imshow(np.clip(A[zona], 0, 255), cmap='gray', vmin=0, vmax=255,
                        interpolation='nearest')
            zoom.set_xticks([]); zoom.set_yticks([])
            for borde in zoom.spines.values():
                borde.set_color('#f59e0b')
    fig.suptitle(etiquetas[nombre] + f' - zoom: rows {r}:{min(r+128,h)}, columns {c}:{min(c+128,w)}')
    plt.savefig(f'p6_dct/resultados/umbrales_{nombre}.png', dpi=150)

# Check the procedure with two inputs whose results are known.
controles = {
    'constante': np.full((8, 8), 128.0, dtype=np.float64),
    'todos los píxeles a cero': np.zeros((8, 8), dtype=np.float64),
}
for nombre, img_control in controles.items():
    X_control = dctn(img_control, norm='ortho')
    db_control = magnitud_db(X_control)
    print(f'\nControl: {etiquetas[nombre]}')
    # isneginf detects -inf; all checks whether it occurs at every position.
    print('All magnitudes are -inf:', np.all(np.isneginf(db_control)))
    print(f"{'Threshold dB':>12} | {'Nonzero coeff. (%)':>20} | {'RMSE (Yq levels)':>20}")
    for umbral in [-120, -90, -60, -30]:
        # Repeat the steps used for the reference and captured images.
        Xu = X_control.copy()
        Xu[db_control < umbral] = 0
        rec_control = idctn(Xu, norm='ortho')
        pct = 100.0 * np.count_nonzero(Xu) / Xu.size
        error = rmse(img_control, rec_control)
        print(f'{umbral:12d} | {pct:20.4f} | {error:20.3e}')

plt.show()

