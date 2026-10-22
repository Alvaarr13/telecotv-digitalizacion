import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from p4_color.conversion import rgb_to_ycrcb, ycrcb_to_rgb
from p5_muestreo.muestreo import submuestrear, reconstruir, rmse, psnr

img = np.array(Image.open('p5_muestreo/datos/detalle_cromatico.png').convert('RGB'))
Yq, Crq, Cbq = rgb_to_ycrcb(img)
Cr = submuestrear(Crq, '420')
Cb = submuestrear(Cbq, '420')
metodos = {'nearest': cv2.INTER_NEAREST,
           'bilineal': cv2.INTER_LINEAR,
           'bicubica': cv2.INTER_CUBIC}
imagenes = {'Original': img}
for nombre, metodo in metodos.items():
    Cr_rec = reconstruir(Cr, Yq.shape, metodo)
    Cb_rec = reconstruir(Cb, Yq.shape, metodo)
    img_rec = ycrcb_to_rgb(Yq, Cr_rec, Cb_rec)
    print(f'{nombre:9s} RMSE={rmse(img,img_rec):.3f}  PSNR={psnr(img,img_rec):.2f} dB')
    imagenes[nombre] = img_rec

# Recortes dentro de las franjas de 1, 2, 4 y 8 píxeles de grosor.
# Cada zona utiliza las mismas coordenadas en las cuatro imágenes.
zonas = [('1 px', 190), ('2 px', 302), ('4 px', 414), ('8 px', 526)]
x0 = 760
fig, ax = plt.subplots(4, 4, figsize=(10, 7))
for fila, (grosor, y0) in enumerate(zonas):
    for columna, (nombre, imagen) in enumerate(imagenes.items()):
        recorte = imagen[y0:y0 + 48, x0:x0 + 64]
        ax[fila, columna].imshow(recorte, interpolation='nearest')
        ax[fila, columna].set_xticks([])
        ax[fila, columna].set_yticks([])
        if fila == 0:
            ax[fila, columna].set_title(nombre)
    ax[fila, 0].set_ylabel(grosor)
fig.suptitle('Franjas de distinto grosor | Original y reconstrucciones 4:2:0')
plt.tight_layout()
plt.savefig('p5_muestreo/resultados/interpolacion.png', dpi=120)

img = np.array(Image.open('p5_muestreo/datos/escena_varillas.png').convert('RGB'))
Yq, Crq, Cbq = rgb_to_ycrcb(img)
Cr = submuestrear(Crq, '420')
Cb = submuestrear(Cbq, '420')
metodos = {'nearest': cv2.INTER_NEAREST,
           'bilineal': cv2.INTER_LINEAR,
           'bicubica': cv2.INTER_CUBIC}
imagenes = {'Original': img}
for nombre, metodo in metodos.items():
    Cr_rec = reconstruir(Cr, Yq.shape, metodo)
    Cb_rec = reconstruir(Cb, Yq.shape, metodo)
    img_rec = ycrcb_to_rgb(Yq, Cr_rec, Cb_rec)
    print(f'{nombre:9s} RMSE={rmse(img,img_rec):.3f}  PSNR={psnr(img,img_rec):.2f} dB')
    imagenes[nombre] = img_rec

fig, ax = plt.subplots(1, 4, figsize=(12, 3.5))
for a, (nombre, imagen) in zip(ax, imagenes.items()):
    a.imshow(imagen[300:428, 1200:1328], interpolation='nearest')
    a.set_title(nombre)
    a.axis('off')
fig.suptitle('Varillas | Mismo recorte del original y las reconstrucciones 4:2:0')
plt.tight_layout()
plt.savefig('p5_muestreo/resultados/interpolacion_varillas.png', dpi=100)

# Tiempo medio de reconstruir las dos componentes de color.
from time import perf_counter
repeticiones = 100
for nombre, metodo in metodos.items():
    # Calentamiento antes de medir.
    for _ in range(5):
        reconstruir(Cr, Yq.shape, metodo)
        reconstruir(Cb, Yq.shape, metodo)
    inicio = perf_counter()
    for _ in range(repeticiones):
        reconstruir(Cr, Yq.shape, metodo)
        reconstruir(Cb, Yq.shape, metodo)
    tiempo_ms = (perf_counter() - inicio) * 1000 / repeticiones
    img_rec = imagenes[nombre]
    print(f'{nombre:9s} RMSE={rmse(img,img_rec):.3f}  '
          f'PSNR={psnr(img,img_rec):.2f} dB  tiempo={tiempo_ms:.3f} ms')
    
plt.show()