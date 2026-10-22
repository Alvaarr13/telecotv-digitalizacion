import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from p4_color.conversion import rgb_to_ycrcb, ycrcb_to_rgb
from p5_muestreo.muestreo import (rgb_to_ycrcb_422, rgb_to_ycrcb_420, ycrcb_422_to_rgb, ycrcb_420_to_rgb,
                                   rmse, psnr)

img = np.array(Image.open('p4_color/datos/escena_referencia.png').convert('RGB'))
Yq, Crq, Cbq = rgb_to_ycrcb(img)
rec444 = ycrcb_to_rgb(Yq, Crq, Cbq)
print(f'4:4:4 base  RMSE={rmse(img, rec444):.3f}')

Yq, Cr, Cb = rgb_to_ycrcb_422(img)
rec422 = ycrcb_422_to_rgb(Yq, Cr, Cb)
Yq, Cr, Cb = rgb_to_ycrcb_420(img)
rec420 = ycrcb_420_to_rgb(Yq, Cr, Cb)
print(f'Dimensiones 4:2:0 (alto, ancho): Yq={Yq.shape}, Cr={Cr.shape}, Cb={Cb.shape}')

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, imagen, titulo in zip(ax, [img, rec422, rec420],
                             ['Entrada RGB', '4:2:2', '4:2:0']):
    a.imshow(imagen); a.set_title(titulo); a.axis('off')
plt.tight_layout()
plt.savefig('p5_muestreo/resultados/muestreo_escena.png', dpi=100)

print(f'4:2:2  RMSE={rmse(img, rec422):.3f}  PSNR={psnr(img, rec422):.2f} dB')
print(f'4:2:0  RMSE={rmse(img, rec420):.3f}  PSNR={psnr(img, rec420):.2f} dB')

fig, ax = plt.subplots(3, 3, figsize=(10, 10), squeeze=False)
def mostrar_comparacion(fila, nombre, ruta, y0, x0):
    original = np.array(Image.open(ruta).convert('RGB'))
    Y, Cr, Cb = rgb_to_ycrcb_422(original)
    rec422 = ycrcb_422_to_rgb(Y, Cr, Cb)
    Y, Cr, Cb = rgb_to_ycrcb_420(original)
    rec420 = ycrcb_420_to_rgb(Y, Cr, Cb)

    imagenes = [original, rec422, rec420]
    titulos = ['Original',
               f'4:2:2 | RMSE = {rmse(original, rec422):.3f}',
               f'4:2:0 | RMSE = {rmse(original, rec420):.3f}']
    for columna, (imagen, titulo) in enumerate(zip(imagenes, titulos)):
        recorte = imagen[y0:y0 + 96, x0:x0 + 96]
        ax[fila, columna].imshow(recorte, interpolation='nearest')
        ax[fila, columna].set_title(titulo, fontsize=10)
        ax[fila, columna].set_xticks([])
        ax[fila, columna].set_yticks([])
    ax[fila, 0].set_ylabel(nombre, fontsize=11)

mostrar_comparacion(0, 'Barras', 'p4_color/datos/barras_referencia.png', 300, 440)
mostrar_comparacion(1, 'Escena', 'p4_color/datos/escena_referencia.png', 300, 740)
mostrar_comparacion(2, 'Detalle cromático', 'p5_muestreo/datos/detalle_cromatico.png', 170, 760)
fig.suptitle('Mismo recorte por fila | RMSE calculado sobre la imagen completa', fontsize=11)
plt.tight_layout()
plt.savefig('p5_muestreo/resultados/comparacion_formatos.png', dpi=120)
plt.show()