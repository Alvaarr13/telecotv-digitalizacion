"""P8: continuidad de la bobina TelecoTV y cuantificación de P7."""
import numpy as np

PESOS_INTER = np.full((8, 8), 16.0, dtype=np.float64)
FACTORES = (1, 2, 4, 10, 30)
VENTANAS = (3, 7, 15, 31)
# Primer píxel (fila, columna), desde cero, sobre la referencia 1920x1080.
BLOQUES = {'vehiculo': (688, 1424), 'fondo': (464, 832), 'pavimento': (864, 640)}
FOTOGRAMAS = (48, 49)
FPS = 25
