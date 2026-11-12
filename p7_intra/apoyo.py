"""Carga, iteración y figuras preparadas para P7; no resuelve la cuantificación."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from p4_color.conversion import rgb_to_ycrcb
from p7_intra.parametros import BLOQUES, FACTORES, ORDEN

SALIDA = Path('p7_intra/resultados')
SALIDA.mkdir(parents=True, exist_ok=True)

def cargar_y(ruta):
    rgb = np.array(Image.open(ruta).convert('RGB'))
    Yq, _, _ = rgb_to_ycrcb(rgb)
    return Yq.astype(np.float64)

def extender(imagen):
    if imagen.ndim != 2 or imagen.dtype != np.float64 or 0 in imagen.shape:
        raise ValueError('Se necesita una imagen 2D float64 no vacía')
    if not np.isfinite(imagen).all():
        raise ValueError('La imagen debe contener valores finitos')
    h, w = imagen.shape
    return np.pad(imagen, ((0, (-h) % 8), (0, (-w) % 8)), mode='edge'), (h, w)

def iter_bloques(imagen):
    # La imagen ya debe tener dimensiones múltiplo de 8.
    for fila in range(0, imagen.shape[0], 8):
        for columna in range(0, imagen.shape[1], 8):
            yield fila, columna, imagen[fila:fila+8, columna:columna+8]

def dct_imagen(imagen, T):
    salida = np.empty_like(imagen)
    for fila, columna, bloque in iter_bloques(imagen):
        salida[fila:fila+8, columna:columna+8] = T @ bloque @ T.T
    return salida

def seleccion(imagen):
    return {nombre: imagen[f:f+8, c:c+8] for nombre, (f, c) in BLOQUES.items()}

def guardar_tabla(nombre, cabecera, filas):
    with (SALIDA / (nombre+'.csv')).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(cabecera); writer.writerows(filas)

def figura_dct(imagen, sin_centrar, centrada):
    # Las dos vistas generales comparten escala; el detalle usa otra escala común.
    limite = max(float(np.abs(sin_centrar).max()), float(np.abs(centrada).max()), 1e-12)
    fig, ax = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    fig.suptitle('DCT por bloques: comparación antes y después de centrar', fontsize=15)
    ax[0, 0].imshow(imagen, cmap='gray', vmin=0, vmax=255)
    ax[0, 0].add_patch(Rectangle((-.5, -.5), 8, 8, fill=False, edgecolor='#f59e0b', linewidth=2))
    ax[0, 0].plot(3.5, 3.5, 'o', ms=17, mfc='none', mec='#f59e0b')
    ax[0, 0].set_title('Entrada Yq; bloque ampliado en naranja')
    for a, X, titulo in zip(ax[0, 1:], [sin_centrar, centrada],
                            ['Sin centrar', 'Tras restar 128']):
        im = a.imshow(X, cmap='coolwarm', vmin=-limite, vmax=limite, interpolation='nearest')
        a.set_title(f'{titulo} — vista general\nMáximo |coeficiente| = {np.abs(X).max():.1f}')
    fig.colorbar(im, ax=list(ax[0, 1:]), shrink=.8, label='Coeficientes DCT; escala común')
    for a in ax[0]:
        a.set_xlabel('Columna de la imagen'); a.set_ylabel('Fila de la imagen')

    # Representar los intervalos con signo, sin confundir máximo con máximo absoluto.
    a = ax[1, 0]
    colores = ['#b45309', '#4338ca']
    for y, X, color in zip([1, 0], [sin_centrar, centrada], colores):
        lo, hi = float(X.min()), float(X.max())
        a.plot([lo, hi], [y, y], color=color, linewidth=5, solid_capstyle='round')
        a.scatter([lo, hi], [y, y], color=color, zorder=3)
        a.annotate(f'{lo:.1f}', (lo, y), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
        a.annotate(f'{hi:.1f}', (hi, y), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
    a.axvline(0, color='#64748b', linestyle='--', linewidth=1)
    a.set(xlim=(-limite*1.16, limite*1.16), ylim=(-.6, 1.65),
          yticks=[0, 1], yticklabels=['Centrada', 'Sin centrar'],
          xlabel='Valor del coeficiente DCT', title='Mínimo y máximo de toda la DCT')
    a.grid(axis='x', alpha=.2)

    # Ampliar exactamente el mismo bloque para ver la posición DC.
    bloques = [sin_centrar[:8, :8], centrada[:8, :8]]
    limite_b = max(float(np.abs(b).max()) for b in bloques)
    limite_b = max(limite_b, 1e-12)
    for a, X, titulo in zip(ax[1, 1:], bloques, ['Sin centrar', 'Centrada']):
        im = a.imshow(X, cmap='coolwarm', vmin=-limite_b, vmax=limite_b, interpolation='nearest')
        a.add_patch(Rectangle((-.5, -.5), 1, 1, fill=False, edgecolor='#111827', linewidth=2))
        a.set_title(f'{titulo}: bloque [0:8, 0:8]\nDC en (0,0) = {X[0,0]:.1f}')
        a.set(xticks=range(8), yticks=range(8), xlabel='Frecuencia horizontal v', ylabel='Frecuencia vertical u')
    fig.colorbar(im, ax=list(ax[1, 1:]), shrink=.8, label='Coeficientes del bloque; escala común')
    fig.savefig(SALIDA/'dct_imagen.png', dpi=140)
    plt.show()

def figura_bloques(imagen, bloques, coeficientes):
    fig, ax = plt.subplots(4, 3, figsize=(11, 12), layout='constrained')
    limite = max(np.abs(X).max() for X in coeficientes.values())
    for i, (nombre, b) in enumerate(bloques.items()):
        f, c = BLOQUES[nombre]
        ax[i, 0].imshow(imagen, cmap='gray', vmin=0, vmax=255)
        ax[i, 0].add_patch(Rectangle((c-.5, f-.5), 8, 8, fill=False, edgecolor='red', linewidth=1))
        ax[i, 0].plot(c+3.5, f+3.5, 'o', ms=18, mfc='none', mec='red')
        ax[i, 0].set_title(f'{nombre}: fila {f}, columna {c}')
        ax[i, 1].imshow(b, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax[i, 1].set_title('Bloque 8x8; escala 0-255')
        im = ax[i, 2].imshow(coeficientes[nombre], cmap='coolwarm', vmin=-limite, vmax=limite, interpolation='nearest')
        ax[i, 2].set_title('DCT centrada; escala común')
        fig.colorbar(im, ax=ax[i, 2], shrink=.65)
    fig.savefig(SALIDA/'cuatro_bloques.png', dpi=140)
    plt.show()

def figura_zigzag(indices):
    orden = np.empty(64, dtype=int); orden[ORDEN] = np.arange(64)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
    ax[0].imshow(orden.reshape(8, 8), cmap='Blues')
    for f in range(8):
        for c in range(8):
            ax[0].text(c, f, str(orden.reshape(8, 8)[f, c]), ha='center', va='center', color='black')
    ax[0].set_title('Orden de lectura: posiciones 0-63')
    v = indices.ravel()[ORDEN]
    ax[1].stem(np.arange(64), v)
    ax[1].set(xlabel='Posición zigzag', ylabel='Índice AC / DC conservado', title='Ejemplo: vegetación, FC=10')
    fig.savefig(SALIDA/'zigzag.png', dpi=130)
    plt.show()

def figura_reconstrucciones(nombre, original, reconstrucciones):
    fig, ax = plt.subplots(2, 5, figsize=(13, 6), layout='constrained')
    limite = max(float(np.abs(original-r).max()) for r in reconstrucciones)
    limite = max(limite, 1e-12)
    ax[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax[0, 0].set_title(nombre+' - entrada'); ax[1, 0].axis('off')
    for j, (fc, rec) in enumerate(zip(FACTORES, reconstrucciones), start=1):
        # Solo imshow limita el dibujo a 0-255; rec no se altera.
        ax[0, j].imshow(rec, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax[0, j].set_title(f'FC={fc}')
        im = ax[1, j].imshow(np.abs(original-rec), cmap='magma', vmin=0, vmax=limite, interpolation='nearest')
        ax[1, j].set_title('|error| en niveles')
    fig.colorbar(im, ax=list(ax[1, 1:]), shrink=.7)
    fig.savefig(SALIDA/('bloque_'+nombre.replace(' ', '_')+'.png'), dpi=130)
    plt.show()

def figura_imagen(nombre, original, reconstrucciones):
    # Seleccionar aproximadamente la misma zona de la escena en ambas entradas.
    # Las fracciones corresponden al detalle [704:800, 400:496] de la referencia.
    # Se supone un encuadre similar tras retirar los márgenes externos de la captura.
    # El recorte conserva la resolución original: no se redimensiona ninguna entrada.
    h, w = original.shape
    f = min(round(h * 704 / 1080), h-1)
    c = min(round(w * 400 / 1920), w-1)
    fin_f = min(h, max(f+1, round(h * 800 / 1080)))
    fin_c = min(w, max(c+1, round(w * 496 / 1920)))
    sl = (slice(f, fin_f), slice(c, fin_c))
    fig, ax = plt.subplots(3, 5, figsize=(15, 9), layout='constrained')
    ax[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
    ax[0, 0].set_title(nombre+' - entrada')
    ax[0, 0].add_patch(Rectangle((c-.5, f-.5), fin_c-c, fin_f-f, fill=False, edgecolor='red'))
    ax[1, 0].imshow(original[sl], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax[1, 0].set_title(f'Detalle [{f}:{sl[0].stop},{c}:{sl[1].stop}]')
    ax[2, 0].axis('off')
    limite = max(float(np.abs(original-r).max()) for r in reconstrucciones)
    for j, (fc, rec) in enumerate(zip(FACTORES, reconstrucciones), start=1):
        ax[0, j].imshow(rec, cmap='gray', vmin=0, vmax=255); ax[0, j].set_title(f'FC={fc}')
        ax[1, j].imshow(rec[sl], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        im = ax[2, j].imshow(np.abs(original-rec)[sl], cmap='magma', vmin=0, vmax=max(limite,1e-12), interpolation='nearest')
        ax[2, j].set_title('|error|; escala común')
    for a in ax.flat: a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=list(ax[2, 1:]), shrink=.7, label='Niveles de Yq')
    fig.savefig(SALIDA/('intra_'+nombre+'.png'), dpi=130)
    plt.show()
