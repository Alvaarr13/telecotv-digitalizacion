"""H.264 del tramo propio: 25 imágenes I frente a una I seguida de 24 P."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from p8_inter.apoyo import SALIDA, guardar_tabla
from p8_inter.parametros import FPS
from p5_muestreo.muestreo import rmse

def ejecutar(args):
    p=subprocess.run(args,capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-1800:])
    return p.stdout

def main():
    ffmpeg,ffprobe=shutil.which('ffmpeg'),shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:raise RuntimeError('Revisar FFmpeg y ffprobe de P4 y reiniciar la terminal')
    with np.load('p8_inter/datos/captura_tramo.npz',allow_pickle=False) as archivo:Y=archivo['Yq']
    if Y.dtype!=np.uint8 or Y.ndim!=3 or Y.shape[0]!=FPS:
        raise ValueError('Volver a ejecutar preparar_captura: se necesitan 25 fotogramas Yq')
    n,h,w=Y.shape
    # yuv420p requiere tamaños pares: repetir solo el último píxel si hace falta.
    # La misma extensión alimenta ambas codificaciones y se retira al medir.
    extendida=np.pad(Y,((0,0),(0,h%2),(0,w%2)),mode='edge')
    he,we=extendida.shape[1:]
    version=ejecutar([ffmpeg,'-version']).decode().splitlines()[0]
    (SALIDA/'version_ffmpeg.txt').write_text(version+'\n',encoding='utf-8')
    condiciones=dict(version=version,entrada='Yq de la captura propia, crominancias a 128',
        fps=FPS,frames=n,forma_original=[h,w],forma_codificada=[he,we],QP=28,ipratio=1,preset='medium',B=0,
        salida='flujo elemental H.264, sin audio ni contenedor',RMSE='sobre todos los fotogramas, área útil original')
    (SALIDA/'condiciones_ffmpeg.json').write_text(json.dumps(condiciones,ensure_ascii=False,indent=2),encoding='utf-8')
    filas=[];reconstrucciones={};tipos={}
    with tempfile.TemporaryDirectory(prefix='telecotv_p8_') as tmp:
        entrada=Path(tmp)/'entrada.yuv'
        with entrada.open('wb') as f:
            for y in extendida:
                f.write(y.tobytes());f.write(bytes([128])*(he*we//2))
        for nombre,gop in [('solo_I',1),('I_P',25)]:
            salida=SALIDA/f'captura_{nombre}.h264'
            ejecutar([ffmpeg,'-v','error','-f','rawvideo','-pixel_format','yuv420p','-video_size',f'{we}x{he}',
                '-framerate',str(FPS),'-i',str(entrada),'-frames:v',str(n),'-c:v','libx264','-preset','medium',
                '-qp','28','-g',str(gop),'-bf','0','-x264-params',f'scenecut=0:keyint={gop}:min-keyint={gop}:ipratio=1',
                '-y',str(salida)])
            info=json.loads(ejecutar([ffprobe,'-v','error','-select_streams','v:0','-show_entries',
                'frame=pict_type,width,height','-of','json',str(salida)]))['frames']
            tipos[nombre]=[f['pict_type'] for f in info]
            esperado=['I']*n if gop==1 else ['I']+['P']*(n-1)
            if tipos[nombre]!=esperado or any((f['height'],f['width'])!=(he,we) for f in info):
                raise RuntimeError('Los tipos o dimensiones no coinciden con la prueba prevista')
            raw=ejecutar([ffmpeg,'-v','error','-i',str(salida),'-frames:v',str(n),'-pix_fmt','yuv420p','-f','rawvideo','-'])
            tam=he*we*3//2
            if len(raw)!=n*tam:raise RuntimeError('Decodificación incompleta')
            rec=np.stack([np.frombuffer(raw,dtype=np.uint8,count=he*we,offset=k*tam).reshape(he,we)[:h,:w] for k in range(n)])
            reconstrucciones[nombre]=rec
            size=salida.stat().st_size;error=float(rmse(Y,rec));kbps=size*8/(n/FPS)/1000
            filas.append((nombre,size,kbps,error,tipos[nombre].count('I'),tipos[nombre].count('P')))
            print(nombre,f'{size} bytes, {kbps:.3f} kbit/s, RMSE={error:.4f}', ''.join(tipos[nombre]))
    guardar_tabla('ffmpeg_temporal',['modo','bytes','kbit_s','RMSE_Yq','I','P'],filas)
    (SALIDA/'tipos_ffmpeg.json').write_text(json.dumps(tipos,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(1,2,figsize=(11,2.2),layout='constrained')
    for a,(nombre,ts) in zip(ax,tipos.items()):
        a.bar(np.arange(n),[1]*n,color=['#6366f1' if t=='I' else '#22c55e' for t in ts])
        for k,t in enumerate(ts):a.text(k,.5,t,ha='center',va='center',fontsize=8,color='white')
        a.set(title=nombre,xlabel='Índice de fotograma (0–24)',yticks=[],xticks=[0,6,12,18,24])
    fig.savefig(SALIDA/'tipos_ffmpeg.png',dpi=140);plt.show()
    # Un mismo fotograma y detalle; errores con escala común, sin amplificar métricas.
    k=12;sl=(slice(int(h*.5),max(int(h*.5)+1,int(h*.82))),slice(int(w*.45),max(int(w*.45)+1,int(w*.9))))
    limite=max(1,max(np.abs(Y.astype(float)-r.astype(float)).max() for r in reconstrucciones.values()))
    fig,ax=plt.subplots(2,3,figsize=(10,5),layout='constrained')
    ax[0,0].imshow(Y[k][sl],cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax[0,0].set_title('Entrada Yq: fotograma 12');ax[1,0].axis('off')
    for j,(nombre,rec) in enumerate(reconstrucciones.items(),1):
        ax[0,j].imshow(rec[k][sl],cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax[0,j].set_title(nombre)
        im=ax[1,j].imshow(np.abs(Y[k].astype(float)-rec[k].astype(float))[sl],cmap='magma',vmin=0,vmax=limite,interpolation='nearest')
        ax[1,j].set_title('|Entrada − decodificada|')
    for a in ax.flat:a.set_xticks([]);a.set_yticks([])
    fig.colorbar(im,ax=list(ax[1,1:]),shrink=.7,label='Niveles Yq; escala común')
    fig.savefig(SALIDA/'detalle_ffmpeg.png',dpi=140);plt.show()
    print('La igualdad de QP no obliga a obtener el mismo RMSE. Comparar tamaño y calidad conjuntamente.')

if __name__=='__main__':main()
