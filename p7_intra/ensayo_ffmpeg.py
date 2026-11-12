"""Ensayo guiado de un solo fotograma I; bytes reales del flujo H.264."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from p7_intra.apoyo import cargar_y, guardar_tabla, SALIDA
from p5_muestreo.muestreo import rmse


def ejecutar(argumentos):
    p = subprocess.run(argumentos, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr.strip() or 'FFmpeg/ffprobe no ha completado la operación')
    return p.stdout


def main():
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('No se encuentra FFmpeg/ffprobe. Revisar la instalación de P4 y reiniciar el terminal de VS Code.')
    imagen = cargar_y('p4_color/datos/escena_referencia.png')
    y = imagen.astype(np.uint8)
    h, w = y.shape
    if h % 2 or w % 2:
        raise ValueError('La referencia de este ensayo debe tener dimensiones pares')
    # Se conserva exactamente Yq de P4 y se fijan ambas crominancias a 128.
    raw = y.tobytes() + bytes([128]) * (y.size//2)
    rows, reconstrucciones = [], []
    version = ejecutar([ffmpeg, '-version']).splitlines()[0]
    print(version)
    print('Un fotograma Yq, crominancias neutras, preset medium, sin audio; salida H.264 elemental.')
    print(f'{"QP":>3} {"tipo":>5} {"bytes":>12} {"RMSE Y":>12}')
    with tempfile.TemporaryDirectory(prefix='telecotv_p7_') as tmp:
        entrada = Path(tmp)/'entrada.yuv'; entrada.write_bytes(raw)
        for qp in (18, 28, 38):
            salida = SALIDA/f'fotograma_qp{qp}.h264'
            cmd = [ffmpeg, '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'yuv420p',
                   '-video_size', f'{w}x{h}', '-framerate', '25', '-i', str(entrada),
                   '-frames:v', '1', '-c:v', 'libx264', '-preset', 'medium', '-qp', str(qp),
                   '-g', '1', '-bf', '0', '-y', str(salida)]
            ejecutar(cmd)
            info = json.loads(ejecutar([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                                      '-show_entries', 'frame=pict_type,width,height', '-of', 'json', str(salida)]))
            frames = info.get('frames', [])
            if len(frames) != 1 or frames[0]['pict_type'] != 'I' or (frames[0]['height'],frames[0]['width']) != (h,w):
                raise RuntimeError('La salida no contiene el único fotograma I del tamaño esperado')
            decodificada = Path(tmp)/'decodificada.yuv'
            ejecutar([ffmpeg, '-v', 'error', '-i', str(salida), '-frames:v', '1', '-f', 'rawvideo',
                      '-pix_fmt', 'yuv420p', '-y', str(decodificada)])
            rec = np.frombuffer(decodificada.read_bytes(), dtype=np.uint8, count=y.size).reshape(h,w).astype(float)
            error = rmse(imagen, rec)
            rows.append((qp, 'I', salida.stat().st_size, float(error)))
            reconstrucciones.append(rec)
            print(f'{qp:3d} {"I":>5} {salida.stat().st_size:12d} {error:12.4f}')
    guardar_tabla('ffmpeg', ['QP', 'tipo', 'bytes', 'RMSE_Y'], rows)
    (SALIDA/'version_ffmpeg.txt').write_text(version+'\n', encoding='utf-8')
    sl = (slice(680, 808), slice(384, 512))
    fig, ax = plt.subplots(2, 4, figsize=(12, 6), layout='constrained')
    ax[0,0].imshow(imagen[sl], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax[0,0].set_title('Entrada Yq - detalle'); ax[1,0].axis('off')
    limite = max(float(np.abs(imagen-r).max()) for r in reconstrucciones)
    for j, (qp, rec) in enumerate(zip((18,28,38), reconstrucciones), start=1):
        ax[0,j].imshow(rec[sl], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax[0,j].set_title(f'H.264, QP={qp}')
        im=ax[1,j].imshow(np.abs(imagen-rec)[sl], cmap='magma', vmin=0, vmax=limite, interpolation='nearest')
        ax[1,j].set_title('|error| en niveles')
    fig.colorbar(im, ax=list(ax[1,1:]), shrink=.7)
    fig.savefig(SALIDA/'ffmpeg.png', dpi=130)
    plt.show()


if __name__ == '__main__':
    main()
