"""Lectura, recorridos y presentación de P8. No implementa las búsquedas."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from p4_color.conversion import rgb_to_ycrcb
from p8_inter.parametros import BLOQUES, VENTANAS

SALIDA = Path('p8_inter/resultados')
SALIDA.mkdir(parents=True, exist_ok=True)

def entero(valor, nombre, minimo=0):
    if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, (int, np.integer)) or valor < minimo:
        raise ValueError(f'{nombre} debe ser entero >= {minimo}')

def validar_par(C, R, comprobar_finitos=True):
    for imagen in (C, R):
        if not isinstance(imagen, np.ndarray) or imagen.ndim != 2 or imagen.dtype != np.float64:
            raise ValueError('Se necesitan imágenes 2D float64')
        if 0 in imagen.shape or (comprobar_finitos and not np.isfinite(imagen).all()):
            raise ValueError('Se necesitan imágenes no vacías y finitas')
    if C.shape != R.shape:
        raise ValueError('C y R deben tener la misma forma')

def validar_busqueda(C, R, fila, columna, p):
    validar_par(C, R, comprobar_finitos=False)
    for nombre, valor in [('fila', fila), ('columna', columna), ('p', p)]:
        entero(valor, nombre)
    if fila % 16 or columna % 16 or fila >= C.shape[0] or columna >= C.shape[1]:
        raise ValueError('La posición debe ser el primer píxel de un macrobloque de la rejilla')
    # El par completo se valida al cargarlo. Aquí se comprueba solo el área que
    # puede leer esta búsqueda, evitando recorrer toda la imagen por cada bloque.
    bh, bw = min(16, C.shape[0]-fila), min(16, C.shape[1]-columna)
    zona = R[max(0,fila-p):min(R.shape[0],fila+bh+p), max(0,columna-p):min(R.shape[1],columna+bw+p)]
    if not np.isfinite(zona).all() or not np.isfinite(C[fila:fila+bh,columna:columna+bw]).all():
        raise ValueError('La región de búsqueda contiene valores no finitos')

def candidato_valido(R, fila, columna, df, dc, p):
    bh, bw = min(16, R.shape[0]-fila), min(16, R.shape[1]-columna)
    return (abs(df) <= p and abs(dc) <= p and 0 <= fila + df <= R.shape[0] - bh
            and 0 <= columna + dc <= R.shape[1] - bw)

def cargar_par(nombre):
    datos = Path('p8_inter/datos')
    canales = []
    for tipo in ('R', 'C'):
        rgb = np.array(Image.open(datos / f'{nombre}_{tipo}.png').convert('RGB'))
        Yq, _, _ = rgb_to_ycrcb(rgb)
        canales.append(Yq.astype(np.float64))
    R, C = canales
    validar_par(C, R)
    return R, C

def guardar_tabla(nombre, cabecera, filas):
    with (SALIDA / f'{nombre}.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(cabecera); w.writerows(filas)

def figura_par(R, C, nombre_par='referencia'):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for a, imagen, titulo in zip(ax, (R, C), ('R: referencia común original', 'C: imagen actual')):
        a.imshow(imagen, cmap='gray', vmin=0, vmax=255)
        a.set(title=titulo, xlabel='Columna', ylabel='Fila')
        for nombre, (f, c) in (BLOQUES.items() if nombre_par == 'referencia' else []):
            a.add_patch(Rectangle((c-.5, f-.5), 16, 16, fill=False, edgecolor='red'))
            a.text(c, f-3, nombre, color='red', fontsize=8)
    fig.savefig(SALIDA/f'par_{nombre_par}.png', dpi=130); plt.show()
    if nombre_par != 'referencia': return
    fig, ax = plt.subplots(3, 2, figsize=(6, 7), layout='constrained')
    for i, (nombre, (f, c)) in enumerate(BLOQUES.items()):
        for j, imagen in enumerate((C, R)):
            ax[i, j].imshow(imagen[f:f+16, c:c+16], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax[i, j].set_title(f'{nombre}: {"C" if j == 0 else "R sin desplazar"}')
    fig.savefig(SALIDA/'bloques_iniciales.png', dpi=130); plt.show()

def comparar_bloques(C, R, buscar, nombre_metodo):
    from p5_muestreo.muestreo import rmse
    filas, resultados = [], {}
    for nombre, (f, c) in BLOQUES.items():
        for p in VENTANAS:
            df, dc, error_prediccion, mae, candidatos = buscar(C, R, f, c, p)
            error = rmse(C[f:f+16, c:c+16], R[f+df:f+df+16, c+dc:c+dc+16])
            filas.append((nombre, p, df, dc, mae, error, candidatos))
            resultados[nombre, p] = (R[f+df:f+df+16, c+dc:c+dc+16], error_prediccion, df, dc, mae, error)
            print(f'{nombre:9s} p={p:2d} df={df:3d} dc={dc:3d} MAE={mae:8.4f} RMSE={error:8.4f} candidatos={candidatos}')
    guardar_tabla(f'bloques_{nombre_metodo}', ['bloque','p','df','dc','MAE','RMSE','candidatos'], filas)
    etiquetas = {'vehiculo': 'Vehículo', 'fondo': 'Fondo', 'pavimento': 'Pavimento'}
    for nombre, (f, c) in BLOQUES.items():
        fig = plt.figure(figsize=(13.5, 5.8), layout='constrained')
        rejilla = fig.add_gridspec(2, 6, width_ratios=[1.15, 1, 1, 1, 1, .07])
        fig.suptitle(f'{nombre_metodo} · {etiquetas[nombre]}: comparación al variar p', fontsize=17, weight='bold')
        original = fig.add_subplot(rejilla[0, 0])
        original.imshow(C[f:f+16, c:c+16], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        original.set_title('Bloque actual de C\nSe mantiene fijo', fontsize=14)
        original.set_xlabel(f'Fila {f}, columna {c}', fontsize=12)
        original.set_xticks([]); original.set_yticks([])
        leyenda = fig.add_subplot(rejilla[1, 0]); leyenda.axis('off')
        leyenda.text(.02, .85, 'El mismo bloque C se compara\ncon los cuatro bloques de R.\n\nRojo: error positivo\nAzul: error negativo\nBlanco: error próximo a 0',
                     transform=leyenda.transAxes, fontsize=12, va='top', linespacing=1.5)
        # Escala común a los cuatro p de esta región. No se comparte entre regiones.
        limite = max(1, max(float(np.abs(resultados[nombre, p][1]).max()) for p in VENTANAS))
        for j, p in enumerate(VENTANAS, 1):
            b, r, df, dc, mae, error = resultados[nombre, p]
            elegido = fig.add_subplot(rejilla[0, j])
            elegido.imshow(b, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            elegido.set_title(f'p = {p}\nBloque elegido en R', fontsize=14)
            elegido.set_xlabel(f'df = {df}, dc = {dc}', fontsize=12)
            error_prediccion_ax = fig.add_subplot(rejilla[1, j])
            im = error_prediccion_ax.imshow(r, cmap='RdBu_r', vmin=-limite, vmax=limite, interpolation='nearest')
            error_prediccion_ax.set_title('Error de predicción\nC − R elegido', fontsize=12)
            error_prediccion_ax.set_xlabel(f'MAE = {mae:.2f}\nRMSE = {error:.2f}', fontsize=12)
            for a in (elegido, error_prediccion_ax): a.set_xticks([]); a.set_yticks([])
        barra = fig.add_subplot(rejilla[1, 5])
        fig.colorbar(im, cax=barra, label='Error de predicción [niveles Yq]', ticks=[-limite, 0, limite])
        fig.supxlabel('Imágenes: escala 0–255. Los errores de predicción comparten escala dentro de cada figura; comprobar la barra al comparar figuras.', fontsize=11)
        fig.savefig(SALIDA/f'bloques_{nombre_metodo}_{nombre}.png', dpi=150)
        plt.show()

def estimar_campo(C, R, p, buscar, mostrar_progreso=True, mostrar_visual=True):
    from time import perf_counter
    validar_par(C, R); entero(p, 'p')
    forma = ((C.shape[0]+15)//16, (C.shape[1]+15)//16)
    df = np.empty(forma, dtype=int); dc = np.empty(forma, dtype=int)
    cuentas = np.empty(forma, dtype=int)
    total = forma[0]*forma[1]
    inicio = ultimo_aviso = perf_counter()
    metodo = getattr(buscar, '__name__', 'busqueda').replace('estim_mov_', '').upper()
    if mostrar_progreso:
        print(f'{metodo}: imagen de {R.shape[1]}×{R.shape[0]} píxeles, p={p}. '
              f'Procesando {total} macrobloques...', flush=True)
        print(f'  0.0 % · 0/{total} bloques', flush=True)
    vista = None
    if mostrar_visual:
        from p8_inter.progreso_visual import ProgresoVisual
        vista = ProgresoVisual(C, R, p, metodo)
    try:
        for i in range(forma[0]):
            for j in range(forma[1]):
                df[i,j], dc[i,j], _, mae, cuentas[i,j] = buscar(C, R, 16*i, 16*j, p)
                hechos = i*forma[1]+j+1
                if vista is not None:
                    vista.registrar(16*i, 16*j, int(df[i,j]), int(dc[i,j]),
                                    float(mae), int(cuentas[i,j]), hechos)
                if mostrar_progreso:
                    ahora = perf_counter()
                    if ahora-ultimo_aviso >= 1 or hechos == total:
                        print(f'  {100*hechos/total:.1f} % · {hechos}/{total} bloques '
                              f'· {ahora-inicio:.1f} s', flush=True)
                        ultimo_aviso = ahora
    finally:
        if vista is not None:
            vista.cerrar()
    return df, dc, cuentas

def guardar_campo(nombre, df, dc, cuentas):
    np.savez(SALIDA/f'campo_{nombre}.npz', df=df, dc=dc, candidatos=cuentas)

def cargar_campo(nombre):
    with np.load(SALIDA/f'campo_{nombre}.npz', allow_pickle=False) as d:
        return d['df'], d['dc'], d['candidatos']

def figura_predicciones(C, resultados, entrada='referencia'):
    fig, ax = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    limite = max(1, max(np.abs(C-P).max() for _, _, P in resultados.values()))
    for i, (nombre, (df, dc, P)) in enumerate(resultados.items()):
        yy, xx = np.mgrid[0:df.shape[0], 0:df.shape[1]]
        ax[i,0].imshow(C, cmap='gray', vmin=0, vmax=255)
        centros_x = 16*xx + (np.minimum(16,C.shape[1]-16*xx)-1)/2
        centros_y = 16*yy + (np.minimum(16,C.shape[0]-16*yy)-1)/2
        ax[i,0].quiver(centros_x, centros_y, dc, df, angles='xy', scale_units='xy', scale=1, color='red', width=.003)
        ax[i,0].set_title(f'{nombre}: correspondencias C → R')
        ax[i,1].imshow(P, cmap='gray', vmin=0, vmax=255)
        ax[i,1].set_title('Predicción compensada')
        im = ax[i,2].imshow(C-P, cmap='coolwarm', vmin=-limite, vmax=limite)
        ax[i,2].set_title('Error de predicción C − predicción')
        for a in ax[i]: a.set(xlabel='Columna (derecha +)', ylabel='Fila (abajo +)')
    fig.colorbar(im, ax=list(ax[:,2]), shrink=.7, label='Niveles Yq; escala común')
    fig.savefig(SALIDA/f'predicciones_{entrada}.png', dpi=130); plt.show()

def figura_reconstrucciones(nombre, C, reconstrucciones, factores, entrada='referencia'):
    # Ampliar la rueda delantera: mismo recorte para todos los FC de esta entrada.
    h, w = C.shape
    # En imágenes pequeñas, incluir más contexto evita un aumento excesivo.
    lado = min(h, w, max(72, int(round(.09*h))))
    f0 = min(max(0, int(round(.665*h))-lado//2), h-lado)
    c0 = min(max(0, int(round(.775*w))-lado//2), w-lado)
    f1, c1 = f0+lado, c0+lado
    sl = (slice(f0,f1), slice(c0,c1))
    # La escala se calcula sobre el detalle mostrado y se comparte entre sus FC.
    limite = max(1, max(float(np.abs(C-r)[sl].max()) for r in reconstrucciones))
    fig = plt.figure(figsize=(15.2, 8), layout='constrained')
    rejilla = fig.add_gridspec(3, len(factores)+2,
                              width_ratios=[1]*(len(factores)+1)+[.055],
                              height_ratios=[.68, 1, 1])
    fig.suptitle(f'{entrada} · {nombre}: reconstrucción y ampliación de la rueda', fontsize=17, weight='bold')
    for j, imagen in enumerate([C, *reconstrucciones]):
        general = fig.add_subplot(rejilla[0,j])
        general.imshow(imagen, cmap='gray', vmin=0, vmax=255)
        general.set_title('C original' if j == 0 else f'FC = {factores[j-1]}', fontsize=14)
        if j == 0:
            general.add_patch(Rectangle((c0-.5,f0-.5),lado,lado,fill=False,edgecolor='#d62728',lw=1.5))
        detalle = fig.add_subplot(rejilla[1,j])
        detalle.imshow(imagen[sl], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        if j == 0:
            detalle.set_title('Detalle de C', fontsize=13)
            detalle.set_xlabel(f'{lado} × {lado} píxeles', fontsize=11)
        else:
            detalle.set_title(f'Detalle · FC = {factores[j-1]}', fontsize=13)
            error_global = float(np.sqrt(np.mean((C-imagen)**2)))
            detalle.set_xlabel(f'RMSE global = {error_global:.3f}', fontsize=11)
            diferencia = fig.add_subplot(rejilla[2,j])
            im = diferencia.imshow(np.abs(C-imagen)[sl], cmap='magma',vmin=0,vmax=limite,interpolation='nearest')
            diferencia.set_title('Diferencia con C', fontsize=12)
            diferencia.set_xticks([]); diferencia.set_yticks([])
        for ax in (general,detalle): ax.set_xticks([]); ax.set_yticks([])
    explicacion = fig.add_subplot(rejilla[2,0]);explicacion.axis('off')
    explicacion.text(.02,.85,'Cuanto más claro es el color,\nmayor es la diferencia\nrespecto a C.',
                     transform=explicacion.transAxes,va='top',fontsize=11,linespacing=1.5)
    barra = fig.add_subplot(rejilla[2,-1])
    fig.colorbar(im,cax=barra,label='Diferencia [niveles Yq]',ticks=[0,limite/2,limite])
    fig.supxlabel('Las cinco columnas usan la misma escala de color. El RMSE compara las imágenes completas.',fontsize=11)
    fig.savefig(SALIDA/f'reconstrucciones_{entrada}_{nombre}.png',dpi=150)
    plt.show()
    return fig
