from p8_inter.apoyo import cargar_par, figura_par, comparar_bloques
from p8_inter.movimiento import estim_mov_fs

R, C = cargar_par('referencia')
figura_par(R, C)
comparar_bloques(C, R, estim_mov_fs, 'FS')