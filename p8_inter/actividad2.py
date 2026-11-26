from p8_inter.apoyo import cargar_par, comparar_bloques
from p8_inter.movimiento import estim_mov_ls
from p8_inter.visualizacion_ls import figura_recorrido_ls

R, C = cargar_par('referencia')
comparar_bloques(C, R, estim_mov_ls, 'LS')
figura_recorrido_ls(C, R, estim_mov_ls, p=15)