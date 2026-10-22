"""Cierre visual de P5. Reutiliza la sincronización guardada en P4.

python -m p5_muestreo.comparar_captura "ruta/captura.mkv"
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from p4_color.conversion import rgb_to_ycrcb, ycrcb_to_rgb


def comparar(imagen):
    """Mismo muestreo y reconstrucción nearest para Y o para Cr/Cb."""
    Y, Cr, Cb = rgb_to_ycrcb(imagen)
    h, w = Y.shape
    if min(h, w) < 8:
        raise ValueError('Imagen demasiado pequeña / Image too small.')
    resultados = []
    for factor in (2, 8):
        def reducir(c):
            sub = cv2.resize(c, (w // factor, h // factor), interpolation=cv2.INTER_NEAREST)
            return cv2.resize(sub, (w, h), interpolation=cv2.INTER_NEAREST)
        resultados.append((imagen, ycrcb_to_rgb(Y, reducir(Cr), reducir(Cb)),
                           ycrcb_to_rgb(reducir(Y), Cr, Cb)))
    return resultados


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('--english', action='store_true')
    p.add_argument('--no-show', action='store_true', help='Guardar sin abrir la figura / Save without displaying the figure')
    args = p.parse_args()
    try:
        if not args.video.is_file():
            raise ValueError(f'No se encuentra el vídeo / Video not found: {args.video}')
        if not shutil.which('ffmpeg'):
            raise ValueError('Se necesita ffmpeg, como en P4 / ffmpeg is required, as in P4.')
        datos = Path('p4_color/datos/adquisicion.json')
        if not datos.is_file():
            raise ValueError('Falta p4_color/datos/adquisicion.json: recuperar los datos de P4 / Restore P4 data.')
        info = json.loads(datos.read_text(encoding='utf-8'))
        if info.get('video') != args.video.name:
            print(('Note: the filename differs from P4. The saved timing is reused; the recording must be the same.'
                   if args.english else
                   'Aviso: el nombre difiere del de P4. Se reutiliza la sincronización guardada; debe ser la misma grabación.'))
        claqueta = float(info['inicio_claqueta_s'])
        if not np.isfinite(claqueta) or claqueta < 0:
            raise ValueError('Sincronización de P4 no válida / Invalid P4 timing.')
        # Centro del plano 03: 18 s; comienzo de la primera claqueta: 2 s.
        t = claqueta + 16
        proc = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', str(args.video),
            '-map', '0:v:0', '-frames:v', '1', '-c:v', 'png', '-f', 'image2pipe', '-'],
            capture_output=True, check=True)
        bgr = cv2.imdecode(np.frombuffer(proc.stdout, dtype=np.uint8), cv2.IMREAD_COLOR) if proc.stdout else None
        if bgr is None:
            raise ValueError('No hay fotograma en ese instante / No frame at the selected time.')
        imagen = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resultados = comparar(imagen)
        if args.no_show:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        h, w = imagen.shape[:2]
        lado = min(256, h, w)
        y, x = (h - lado) // 2, (w - lado) // 2
        fig, axs = plt.subplots(2, 3, figsize=(11, 9.2))
        for fila, factor in enumerate((2, 8)):
            titulos = (('Original capture', f'Cr and Cb: width and height ÷{factor}', f'Y: width and height ÷{factor}')
                       if args.english else ('Captura original', f'Cr y Cb: ancho y alto ÷{factor}', f'Y: ancho y alto ÷{factor}'))
            for ax, img, titulo in zip(axs[fila], resultados[fila], titulos):
                ax.imshow(img[y:y+lado, x:x+lado], interpolation='nearest')
                ax.set_title(titulo, fontsize=11)
                ax.axis('off')
        fig.suptitle('Captured rods · same crop · nearest reconstruction' if args.english else
                     'Varillas capturadas · mismo recorte · reconstrucción nearest', fontsize=14)
        nota = ('÷8 is an exaggerated experiment, not 4:2:0. Visual effects are compared, not equal data savings.' if args.english else
                '÷8 es una prueba exagerada, no 4:2:0. Se compara el efecto visual, no igual ahorro de datos.')
        fig.text(.5, .02, nota, ha='center', fontsize=10)
        fig.subplots_adjust(left=.025, right=.975, bottom=.07, top=.90, wspace=.05, hspace=.18)
        salida = Path('p5_muestreo/resultados')
        salida.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(salida / 'varillas_capturadas.png'), bgr)
        fig.savefig(salida / 'captura_color_luminancia.png', dpi=150)
        print(salida / 'captura_color_luminancia.png')
        print('Check that the image shows rods. Otherwise, check P4 timing.' if args.english else
              'Comprobar que aparecen varillas. Si no, revisar la sincronización de P4.')
        if not args.no_show:
            plt.show()
        plt.close(fig)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        p.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
