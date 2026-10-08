"""Herramienta proporcionada de adquisición y registro geométrico para P4.

Desde la raíz del repositorio:
  python -m p4_color.captura extraer "ruta/captura.mkv" --claqueta 10.2
  python -m p4_color.captura comparar

--claqueta es el instante, en segundos de TU grabación, en que comienza la
primera claqueta (PLANO 01). No es el comienzo del negro inicial.
No se corrigen niveles ni se promedian fotogramas. Las métricas siempre
requieren comprobar visualmente el registro antes de interpretarlas.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
DATOS = BASE / 'datos'
RESULTADOS = BASE / 'resultados'
W, H = 1920, 1080
MARCO = (97.5, 55.5, 1821.5, 1023.5)


def ejecutar(args):
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode:
        raise ValueError(proc.stderr.decode(errors='replace')[-1800:])
    return proc.stdout


def cargar(nombre):
    # imdecode permite rutas con caracteres no ASCII también en Windows.
    ruta = DATOS / nombre
    if not ruta.is_file():
        raise ValueError(f'No se encuentra {ruta}. Revisa los materiales y la extracción.')
    img = cv2.imdecode(np.frombuffer(ruta.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'No se puede leer {ruta}.')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def guardar(ruta, rgb):
    ok, buf = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f'No se puede codificar {ruta.name}.')
    ruta.write_bytes(buf.tobytes())


def extraer(video, claqueta):
    if not np.isfinite(claqueta) or claqueta < 0:
        raise ValueError('El tiempo de la claqueta debe ser un número positivo o cero.')
    if not video.is_file():
        raise ValueError(f'No se encuentra el vídeo: {video}')
    for herramienta in ('ffmpeg', 'ffprobe'):
        if not shutil.which(herramienta):
            raise ValueError(f'No se encuentra {herramienta}. Revisa su instalación y reinicia la terminal.')
    info = json.loads(ejecutar(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate,pix_fmt,color_space,color_range,field_order,sample_aspect_ratio:format=duration',
        '-of', 'json', str(video)]))
    if not info.get('streams'):
        raise ValueError('El fichero no contiene una pista de vídeo.')
    DATOS.mkdir(parents=True, exist_ok=True)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    # Bobina v1: claqueta 1 a 2 s; centros de barras y escena a 6 y 12 s.
    tiempos = {'claqueta_capturada.png': claqueta + 1,
               'barras_capturadas.png': claqueta + 4,
               'escena_capturada.png': claqueta + 10}
    frames = {}
    for nombre, t in tiempos.items():
        raw = ejecutar(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', str(video),
            '-map', '0:v:0', '-frames:v', '1', '-c:v', 'png', '-f', 'image2pipe', '-'])
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
        if bgr is None:
            raise ValueError(f'No hay fotograma en {t:.2f} s. Revisa la duración y el tiempo de la claqueta.')
        frames[nombre] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for nombre, rgb in frames.items():
        guardar(DATOS / nombre, rgb)
    info.update({'video': video.name, 'inicio_claqueta_s': claqueta,
                 'tiempos_extraidos_s': tiempos, 'bobina': 'v1'})
    (DATOS / 'adquisicion.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (nombre, rgb) in zip(axes, frames.items()):
        ax.imshow(rgb); ax.set_title(nombre.replace('_', ' '), fontsize=9); ax.axis('off')
    fig.tight_layout(); fig.savefig(RESULTADOS / 'extraccion.png', dpi=120); plt.close(fig)
    print(json.dumps(info['streams'][0], ensure_ascii=False, indent=2))
    print('Guardados tres PNG y datos/adquisicion.json. Abre resultados/extraccion.png.')
    print('Comprueba: claqueta PLANO 01, ocho barras y escena del camino con una persona.')
    print('Si no coinciden, corrige --claqueta y repite; no continúes con imágenes equivocadas.')


def matriz_marco(rgb):
    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luminosos = gris > 150
    filas = np.flatnonzero(luminosos.mean(axis=1) > .3)
    cols = np.flatnonzero(luminosos.mean(axis=0) > .3)
    if len(filas) < 2 or len(cols) < 2:
        raise ValueError('No se detecta el marco. Comprueba claqueta_capturada.png y el instante de extracción.')
    def extremos(v):
        return v[v <= v.min() + 6].mean(), v[v >= v.max() - 6].mean()
    x1, x2 = extremos(cols); y1, y2 = extremos(filas)
    if x2 - x1 < rgb.shape[1] * .3 or y2 - y1 < rgb.shape[0] * .3:
        raise ValueError('El marco detectado es demasiado pequeño. Solicita revisión del profesor.')
    sx = (x2 - x1) / (MARCO[2] - MARCO[0])
    sy = (y2 - y1) / (MARCO[3] - MARCO[1])
    return np.array([[sx, 0, x1 - MARCO[0] * sx],
                     [0, sy, y1 - MARCO[1] * sy]], np.float32)


def comparar():
    ref = cargar('escena_referencia.png')
    cap = cargar('escena_capturada.png')
    claqueta = cargar('claqueta_capturada.png')
    if ref.shape != (H, W, 3) or cap.shape != claqueta.shape:
        raise ValueError('Dimensiones inesperadas. Usa las referencias de P4 y una sola grabación sin cambiar el encuadre.')
    M = matriz_marco(claqueta)
    ch, cw = cap.shape[:2]
    ref_peq = cv2.warpAffine(ref, M, (cw, ch), flags=cv2.INTER_LINEAR)
    mascara_peq = cv2.warpAffine(np.full((H, W), 255, np.uint8), M, (cw, ch))
    mascara_peq = cv2.erode(mascara_peq, np.ones((9, 9), np.uint8))
    gris = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (0, 0), 2).astype(np.float32)
    D = np.eye(2, 3, dtype=np.float32)
    # Refinamiento geométrico limitado: no permite una deformación grande
    # para compensar una escena errónea. El control visual sigue siendo necesario.
    try:
        correlacion, D = cv2.findTransformECC(gris(ref_peq), gris(cap), D,
            cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), mascara_peq, 5)
    except cv2.error as exc:
        raise ValueError('No converge la alineación. Comprueba que es el PLANO 02 completo; pide revisión antes de medir.') from exc
    if (correlacion < .7 or np.max(np.abs(D[:, :2] - np.eye(2))) > .15
            or np.max(np.abs(D[:, 2])) > .1 * max(cw, ch)):
        raise ValueError('La alineación no es fiable. Revisa la escena y el marco con el profesor.')
    M = (np.vstack([D, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]]))[:2].astype(np.float32)
    flags = cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP
    alineada = cv2.warpAffine(cap, M, (W, H), flags=flags)
    validos = cv2.warpAffine(np.full((ch, cw), 255, np.uint8), M, (W, H),
                            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) == 255
    validos[:20] = False; validos[-20:] = False; validos[:, :20] = False; validos[:, -20:] = False
    validos = cv2.erode(validos.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    if validos.mean() < .8:
        raise ValueError('La región común es insuficiente; revisa el encuadre.')
    dif = ref.astype(np.float64) - alineada.astype(np.float64)
    rmse = float(np.sqrt(np.mean(dif[validos] ** 2)))
    psnr = float(20 * np.log10(255 / rmse)) if rmse else None
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    guardar(RESULTADOS / 'escena_alineada.png', alineada)
    np.savez_compressed(RESULTADOS / 'comparacion.npz', referencia=ref, captura=alineada, mascara=validos)
    metricas = {'rmse_rgb': rmse, 'psnr_rgb_db': psnr, 'correlacion_ecc': float(correlacion),
        'pixeles_evaluados': int(validos.sum()), 'matriz_referencia_a_captura': M.tolist(),
        'metodo': 'Un fotograma; captura interpolada a 1920x1080 (bicubica); region comun interior; sin correccion de niveles',
        'requiere_comprobacion_visual': True}
    (RESULTADOS / 'metricas_captura.json').write_text(json.dumps(metricas, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    overlay = np.round(.5 * ref + .5 * alineada).astype(np.uint8)
    for ax, im, title in zip(axes.flat, [ref, alineada, overlay],
            ['Referencia digital', 'Captura alineada — sin corregir niveles', 'Superposición 50 %: comprobar los bordes']):
        ax.imshow(im); ax.set_title(title, fontsize=10); ax.axis('off')
    err = np.mean(np.abs(dif), axis=2); err[~validos] = np.nan
    graf = axes[1, 1].imshow(err, cmap='magma', vmin=0, vmax=80)
    axes[1, 1].set_title('Error absoluto medio RGB (0–80 niveles)', fontsize=10)
    axes[1, 1].axis('off'); fig.colorbar(graf, ax=axes[1, 1], fraction=.03)
    fig.suptitle('Comparación de adquisición — validar la superposición antes de aceptar las métricas', fontsize=11)
    fig.tight_layout(); fig.savefig(RESULTADOS / 'comparacion_captura.png', dpi=130); plt.close(fig)
    print(f'RMSE RGB: {rmse:.4f} niveles; PSNR: {psnr:.2f} dB' if psnr is not None else 'RMSE: 0; PSNR: infinito')
    print('Abre resultados/comparacion_captura.png y comprueba persona, camino y árboles.')
    print('Si hay bordes dobles desplazados, no aceptes las métricas: pide revisión.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='accion', required=True)
    ext = sub.add_parser('extraer', help='Extraer claqueta, barras y escena de TU grabación')
    ext.add_argument('video', type=Path)
    ext.add_argument('--claqueta', required=True, type=float)
    sub.add_parser('comparar', help='Alinear la escena y obtener una comparación pendiente de validación visual')
    args = parser.parse_args()
    try:
        if args.accion == 'extraer':
            extraer(args.video, args.claqueta)
        else:
            comparar()
    except (ValueError, OSError) as exc:
        parser.exit(1, f'No se ha completado la operación: {exc}\n')


if __name__ == '__main__':
    main()
