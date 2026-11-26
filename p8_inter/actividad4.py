import numpy as np
from pathlib import Path
from p5_muestreo.muestreo import rmse
from p7_intra.intra import cod_intra, decod_intra, zigzag, ceros_finales
from p7_intra.apoyo import iter_bloques
from p8_inter.parametros import FACTORES, PESOS_INTER
from p8_inter.movimiento import compensar
from p8_inter.apoyo import cargar_par, cargar_campo, guardar_tabla, figura_reconstrucciones

def codificar_error_prediccion(C, prediccion, FC):
    error_prediccion = C - prediccion  # Differences remaining between C and its prediction.
    indices, forma = cod_intra(error_prediccion, FC, pesos=PESOS_INTER, desplazamiento=0)
    recuperado = decod_intra(indices, forma, FC, pesos=PESOS_INTER, desplazamiento=0)
    reconstruida = prediccion + recuperado  # Combine the prediction and recovered error.
    numero_bloques = (indices.shape[0]//8) * (indices.shape[1]//8)
    total_ceros = sum(ceros_finales(zigzag(b)) for _, _, b in iter_bloques(indices))
    porcentaje = (total_ceros / (numero_bloques * 64)) * 100  # Express trailing zeros as a percentage of all coefficients.
    error_reconstruccion = rmse(C, reconstruida)  # RMSE between C and the reconstructed image; use the P5 function.
    return reconstruida, porcentaje, error_reconstruccion

def main():
    filas = []
    for entrada in ('referencia', 'captura'):
        if entrada == 'captura' and not Path('p8_inter/datos/captura_R.png').is_file():
            continue  # Document the issue; no results are fabricated.
        R, C = cargar_par(entrada)
        for nombre in ('FS', 'LS'):
            df, dc, _ = cargar_campo(entrada+'_'+nombre)
            prediccion = compensar(R, df, dc)
            reconstrucciones, admisibles = [], []
            for FC in FACTORES:
                rec, porcentaje, error = codificar_error_prediccion(C, prediccion, FC)
                filas.append((entrada, nombre, FC, porcentaje, error))
                reconstrucciones.append(rec)
                if error <= 2.5: admisibles.append(FC)
                print(entrada, nombre, FC, f'ceros={porcentaje:.4f} % RMSE={error:.4f}')
            print(entrada, nombre, 'Largest FC meeting the RMSE limit:', max(admisibles) if admisibles else 'ninguno')
            figura_reconstrucciones(nombre, C, reconstrucciones, FACTORES, entrada)
    guardar_tabla('reconstruccion', ['entrada','metodo','FC','ceros_finales_pct','RMSE'], filas)

if __name__ == '__main__':
    main()