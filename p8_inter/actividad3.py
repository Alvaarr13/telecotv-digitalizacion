from time import perf_counter
import numpy as np
from p5_muestreo.muestreo import rmse
from p8_inter.movimiento import estim_mov_fs, estim_mov_ls, compensar
from p8_inter.apoyo import cargar_par, estimar_campo, guardar_campo, guardar_tabla, figura_predicciones

filas = []
for entrada in ('referencia', 'captura'):
    if entrada == 'captura':
        from pathlib import Path
        if not Path('p8_inter/datos/captura_R.png').is_file():
            print('Capture pending: the issue must be documented; capture results remain blank')
            continue
    R, C = cargar_par(entrada)
    resultados = {}
    for nombre, buscar in [('FS', estim_mov_fs), ('LS', estim_mov_ls)]:
        inicio = perf_counter()
        df, dc, cuentas = estimar_campo(C, R, 15, buscar)
        segundos = perf_counter() - inicio
        prediccion = compensar(R, df, dc)
        error_prediccion = C - prediccion  # C minus prediccion, retaining levels and sign.
        assert np.allclose(prediccion + error_prediccion, C, atol=1e-12, rtol=0)
        mae = float(np.mean(np.abs(error_prediccion)))
        error = rmse(C, prediccion)
        print(entrada, nombre, 'MAE=', mae, 'RMSE=', error, 'candidatos=', cuentas.sum(), 's=', segundos)
        filas.append((entrada, nombre, mae, error, int(cuentas.sum()), segundos))
        guardar_campo(entrada+'_'+nombre, df, dc, cuentas)
        resultados[nombre] = (df, dc, prediccion)
    figura_predicciones(C, resultados, entrada)
guardar_tabla('prediccion', ['entrada','metodo','MAE','RMSE','candidatos','segundos'], filas)