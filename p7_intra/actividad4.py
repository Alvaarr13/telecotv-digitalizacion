import numpy as np
import matplotlib.pyplot as plt
from p5_muestreo.muestreo import rmse
from p7_intra.intra import cod_intra, decod_intra, zigzag, ceros_finales
from p7_intra.apoyo import cargar_y, iter_bloques, figura_imagen, guardar_tabla
from p7_intra.parametros import FACTORES

referencia = cargar_y('p4_color/datos/escena_referencia.png')
captura = cargar_y('p4_color/datos/escena_capturada.png')
# Reuse the useful-image boundaries checked in P6.
fila_inicial, fila_final = 36, 460
columna_inicial, columna_final = 67, 664
h, w = captura.shape
if not (0 <= fila_inicial < fila_final <= h and 0 <= columna_inicial < columna_final <= w):
    raise ValueError('The useful-image boundaries must be checked')
captura_util = captura[fila_inicial:fila_final, columna_inicial:columna_final]
plt.imshow(captura_util, cmap='gray', vmin=0, vmax=255)
plt.title('Useful image area from P6: boundaries must be checked'); plt.show()
print(f'Useful image area: rows {fila_inicial}:{fila_final}, columns {columna_inicial}:{columna_final}')
imagenes = {'referencia': referencia, 'captura': captura_util}
filas = []
for nombre, imagen in imagenes.items():
    reconstrucciones = []
    print(f'\n{nombre}:')
    print(f'{"FC":>3s} {"blocks":>8s} {"trailing zeros [%]":>18s} {"RMSE Yq":>12s}')
    for FC in FACTORES:
        indices, forma = cod_intra(imagen, FC)
        rec = decod_intra(indices, forma, FC)
        numero_bloques = (indices.shape[0]//8) * (indices.shape[1]//8)
        total_ceros = sum(ceros_finales(zigzag(b)) for _, _, b in iter_bloques(indices))
        porcentaje = 100 * total_ceros / (64 * numero_bloques)
        error = rmse(imagen, rec)
        print(f'{FC:3d} {numero_bloques:8d} {porcentaje:18.4f} {error:12.4f}')
        filas.append((nombre, FC, porcentaje, error))
        reconstrucciones.append(rec)
    figura_imagen(nombre, imagen, reconstrucciones)
guardar_tabla('imagen_completa', ['entrada', 'FC', 'ceros_finales_pct', 'RMSE_Yq'], filas)