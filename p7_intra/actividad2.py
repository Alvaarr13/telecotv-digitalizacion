import numpy as np
from p6_dct.dct import mi_dct
from p7_intra.intra import cuantificar, zigzag, inversa_zigzag, ceros_finales, redondear
from p7_intra.parametros import FACTORES
from p7_intra.apoyo import cargar_y, seleccion, figura_zigzag, guardar_tabla

print('Rounding values halfway between two integers:', redondear(np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])))
prueba = np.arange(64, dtype=np.float64).reshape(8, 8)
assert np.array_equal(inversa_zigzag(zigzag(prueba)), prueba)
assert ceros_finales(np.zeros(64)) == 64
v = np.zeros(64)
v[0] = 1
assert ceros_finales(v) == 63
v[-1] = 1
assert ceros_finales(v) == 0
print('Scan and zero-count checks passed')

bloques = seleccion(cargar_y('p4_color/datos/escena_referencia.png'))
filas = []
print(f'{"block":18s} {"FC":>3s} {"trailing zeros [%]":>18s}')
for nombre, b in bloques.items():
    X = mi_dct(b - 128)
    for FC in FACTORES:
        indices = cuantificar(X, FC)
        porcentaje = 100 * ceros_finales(zigzag(indices)) / 64
        filas.append((nombre, FC, porcentaje))
        print(f'{nombre:18s} {FC:3d} {porcentaje:18.4f}')
    print()
guardar_tabla('ceros_bloques', ['bloque', 'FC', 'ceros_finales_pct'], filas)
figura_zigzag(cuantificar(mi_dct(bloques['vegetacion'] - 128), 10))