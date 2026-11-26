"""Extrae el tramo temporal de la grabación propia de P4, sin nueva adquisición."""
from pathlib import Path
import argparse
from fractions import Fraction
import json
import shutil
import subprocess
import numpy as np
from PIL import Image
from p4_color.conversion import rgb_to_ycrcb
from p8_inter.parametros import FOTOGRAMAS, FPS

def ejecutar(args):
    p = subprocess.run(args, capture_output=True)
    if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace')[-1800:])
    return p.stdout

def preparar(video, filas, columnas, claqueta=None):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise RuntimeError('Recuperar FFmpeg y ffprobe de P4 y reiniciar la terminal')
    if not video.is_file(): raise ValueError(f'No se encuentra la grabación de P4: {video}')
    if claqueta is None:
        ruta = Path('p4_color/datos/adquisicion.json')
        if not ruta.is_file():
            raise ValueError('Recuperar adquisicion.json de P4 o indicar --claqueta con el inicio comprobado de la primera claqueta')
        info = json.loads(ruta.read_text(encoding='utf-8'))
        claqueta = float(info['inicio_claqueta_s'])
        if info.get('video') != video.name:
            raise ValueError('El vídeo no coincide con adquisicion.json. Recuperar la grabación correcta o indicar --claqueta tras comprobarla')
    if not np.isfinite(claqueta) or claqueta < 0: raise ValueError('Inicio de claqueta inválido')
    stream = json.loads(ejecutar(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate','-of','json',str(video)]))['streams'][0]
    h, w = stream['height'], stream['width']
    if abs(float(Fraction(stream['r_frame_rate'])) - FPS) > .01:
        raise ValueError('Esta extracción está preparada para la grabación de P4 a 25 fps. Revisar su configuración con el profesor')
    f0, f1 = filas; c0, c1 = columnas
    if not (0 <= f0 < f1 <= h and 0 <= c0 < c1 <= w): raise ValueError('Revisar los límites propios del área útil de P6/P7')
    # Primera claqueta a 2 s; movimiento a 52 s en la bobina.
    inicio = claqueta + 50 + FOTOGRAMAS[0]/FPS
    raw = ejecutar(['ffmpeg','-v','error','-ss',f'{inicio:.6f}','-i',str(video),
        '-map','0:v:0','-frames:v',str(FPS),'-fps_mode','passthrough','-pix_fmt','rgb24','-f','rawvideo','-'])
    if len(raw) != FPS*h*w*3:
        raise ValueError('Falta el tramo completo de movimiento. Comprobar grabación, claqueta y duración; no fabricar fotogramas')
    frames = np.frombuffer(raw,np.uint8).reshape(FPS,h,w,3)[:,f0:f1,c0:c1].copy()
    if np.array_equal(frames[0],frames[1]):
        raise ValueError('Los dos fotogramas extraídos son idénticos. Revisar el tramo y la cadencia con el profesor')
    datos=Path('p8_inter/datos');datos.mkdir(parents=True,exist_ok=True)
    Image.fromarray(frames[0]).save(datos/'captura_R.png')
    Image.fromarray(frames[1]).save(datos/'captura_C.png')
    Y = np.stack([rgb_to_ycrcb(frame)[0] for frame in frames])
    np.savez_compressed(datos/'captura_tramo.npz',Yq=Y)
    registro=dict(video=str(video.resolve()),inicio_claqueta_s=claqueta,inicio_tramo_extraido_s=inicio,
                  fps=FPS,numero_fotogramas=FPS,filas=filas,columnas=columnas,forma_nativa=[h,w],forma_util=list(Y.shape[1:]))
    (datos/'extraccion_temporal.json').write_text(json.dumps(registro,ensure_ascii=False,indent=2),encoding='utf-8')
    from p8_inter.apoyo import figura_par
    print(json.dumps(registro,ensure_ascii=False,indent=2))
    print('\nA continuación se mostrarán los dos fotogramas extraídos de tu grabación.')
    print('Comprueba que ambos corresponden a la escena de tráfico del PLANO 09 y que el recorte conserva el área útil elegida.')
    print('Fíjate en la posición de los vehículos para comprobar el movimiento entre ambos instantes. Algunas diferencias entre las imágenes pueden deberse al ruido de la grabación.')
    input('\nPulsa Enter para ver las imágenes... ')
    figura_par(Y[0].astype(np.float64),Y[1].astype(np.float64),'captura')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--filas',type=int,nargs=2,required=True,metavar=('INICIO','FIN'))
    p.add_argument('--columnas',type=int,nargs=2,required=True,metavar=('INICIO','FIN'))
    p.add_argument('--claqueta',type=float,help='Solo si falta el registro de P4 o se usa otra grabación comprobada')
    a=p.parse_args();preparar(a.video,a.filas,a.columnas,a.claqueta)

if __name__=='__main__': main()
