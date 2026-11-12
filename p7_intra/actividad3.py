import numpy as np
from p6_dct.dct import mi_dct, mi_idct
from p5_muestreo.muestreo import rmse
from p7_intra.intra import cuantificar, cuantificacion_inversa
from p7_intra.parametros import FACTORES
from p7_intra.apoyo import cargar_y, seleccion, guardar_tabla, figura_reconstrucciones

bloques = seleccion(cargar_y('p4_color/datos/escena_referencia.png'))
filas = []
for nombre, b in bloques.items():
    X = mi_dct(b - 128)
    reconstrucciones = []
    print(f'\n{nombre}:')
    print(f'{"FC":>3s} {"max |X-Xr|":>14s} {"Spatial RMSE":>16s}')
    for FC in FACTORES:
        indices = cuantificar(X, FC)
        Xr = cuantificacion_inversa(indices, FC)
        rec = mi_idct(Xr) + 128            # Apply mi_idct to Xr and restore the levels.
        error_max = np.max(np.abs(X - Xr))             # Maximum absolute coefficient difference.
        error_rmse = rmse(b, rec)            # Call rmse with the original block b and its reconstruction rec.
        
        reconstrucciones.append(rec)
        filas.append((nombre, FC, error_max, error_rmse))
        print(f'{FC:3d} {error_max:14.4f} {error_rmse:16.4f}')
    figura_reconstrucciones(nombre, b, reconstrucciones)
guardar_tabla('error_bloques', ['bloque', 'FC', 'error_max_DCT', 'RMSE'], filas)
# DC may have a fractional part. Do not convert the whole matrix to integers.
b = np.full((8, 8), 150.125, dtype=np.float64)
X = mi_dct(b - 128)
for FC in FACTORES:
    assert cuantificacion_inversa(cuantificar(X, FC), FC)[0, 0] == X[0, 0]
print('DC retains its original value, including any fractional part')