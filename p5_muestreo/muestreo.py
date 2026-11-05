import cv2
import numpy as np
from p4_color.conversion import rgb_to_ycrcb, ycrcb_to_rgb

def submuestrear(C, modo):
    # C: crominancia 2D (alto, ancho)
    h, w = C.shape
    if modo == '422':
        dsize = (w//2, h)       # mitad de columnas
    elif modo == '420':
        dsize = (w//2, h//2)     # mitad de columnas y filas
    else:
        raise ValueError('Usar 422 o 420')
    return cv2.resize(C, dsize, interpolation=cv2.INTER_NEAREST)

def reconstruir(C_sub, shape_destino, metodo=cv2.INTER_NEAREST):
    h, w = shape_destino           # shape = (alto, ancho)
    return cv2.resize(C_sub, (w, h), interpolation=metodo)

def rgb_to_ycrcb_422(imagen_rgb):
    Yq, Crq, Cbq = rgb_to_ycrcb(imagen_rgb)
    return Yq, submuestrear(Crq, '422'), submuestrear(Cbq, '422')

def ycrcb_422_to_rgb(Yq, Cr, Cb, metodo=cv2.INTER_NEAREST):
    Crq = reconstruir(Cr, Yq.shape, metodo)
    Cbq = reconstruir(Cb, Yq.shape, metodo)
    return ycrcb_to_rgb(Yq, Crq, Cbq)

# Repetir el patrón para 4:2:0: completar el modo '420'
def rgb_to_ycrcb_420(imagen_rgb):
    Yq, Crq, Cbq = rgb_to_ycrcb(imagen_rgb)
    return Yq, submuestrear(Crq, '420'), submuestrear(Cbq, '420')

def ycrcb_420_to_rgb(Yq, Cr, Cb, metodo=cv2.INTER_NEAREST):
    Crq = reconstruir(Cr, Yq.shape, metodo)
    Cbq = reconstruir(Cb, Yq.shape, metodo)
    return ycrcb_to_rgb(Yq, Crq, Cbq)

def rmse(orig, recon):
    if orig.shape != recon.shape:
        raise ValueError('Las imagenes deben tener las mismas dimensiones')
    o = orig.astype(np.float64)
    r = recon.astype(np.float64)
    return np.sqrt(np.mean((o - r) ** 2))

def psnr(orig, recon):
    error = rmse(orig, recon)
    if error == 0:
        return float('inf')
    return 20 * np.log10(255.0 / error)

