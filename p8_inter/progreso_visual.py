"""Vista del campo de movimiento mientras se calcula; no implementa búsquedas."""
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class ProgresoVisual:
    """Muestra bloques ya calculados y actualiza la ventana como máximo a 2 Hz."""

    def __init__(self, C, R, p, metodo):
        self.C, self.R, self.p = C, R, p
        self.prediccion = np.full_like(C, np.nan)
        self.total = ((C.shape[0]+15)//16)*((C.shape[1]+15)//16)
        self.inicio = perf_counter()
        self.ultimo = -np.inf
        self.paso = max(1, int(np.ceil(max(C.shape)/700)))
        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 7))
        self.fig.subplots_adjust(left=.055, right=.98, top=.86, bottom=.10, hspace=.43, wspace=.25)
        self.titulo = self.fig.suptitle(f'{metodo} · preparando la búsqueda…', fontsize=16, weight='bold')
        self.metodo = metodo
        h, w = C.shape
        extension = (-.5, w-.5, h-.5, -.5)
        self.axes[0,0].imshow(C[::self.paso,::self.paso], cmap='gray', vmin=0, vmax=255, extent=extension)
        self.axes[0,0].set_title('C: posición del bloque actual')
        self.actual = Rectangle((-.5,-.5),16,16,fill=False,ec='#FFD700',lw=2)
        self.axes[0,0].add_patch(self.actual)
        self.horizontal = self.axes[0,0].axhline(0,color='#FFD700',lw=.6,alpha=.7)
        self.vertical = self.axes[0,0].axvline(0,color='#FFD700',lw=.6,alpha=.7)
        self.ventana = self.axes[0,1].imshow(np.zeros((16,16)),cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        self.axes[0,1].set_title('R: región donde se ha buscado')
        self.sin_desplazar = Rectangle((-.5,-.5),16,16,fill=False,ec='#FFD700',lw=1.5,ls='--')
        self.elegido = Rectangle((-.5,-.5),16,16,fill=False,ec='#00BFFF',lw=2)
        self.axes[0,1].add_patch(self.sin_desplazar); self.axes[0,1].add_patch(self.elegido)
        self.axes[0,1].set_xlabel('Columna de R'); self.axes[0,1].set_ylabel('Fila de R')
        gris = plt.get_cmap('gray').copy(); gris.set_bad('#dce5ed')
        self.formada = self.axes[0,2].imshow(np.ma.masked_invalid(self.prediccion[::self.paso,::self.paso]),
                                           cmap=gris,vmin=0,vmax=255,extent=extension)
        self.axes[0,2].set_title('Predicción que se va formando')
        self.bloques = []
        for ax, titulo in zip(self.axes[1,:2], ('Bloque actual de C', 'Bloque elegido en R')):
            self.bloques.append(ax.imshow(np.zeros((16,16)),cmap='gray',vmin=0,vmax=255,interpolation='nearest'))
            ax.set_title(titulo); ax.set_xticks([]); ax.set_yticks([])
        for ax in (self.axes[0,0],self.axes[0,2]): ax.set_xticks([]); ax.set_yticks([])
        estado = self.axes[1,2]; estado.axis('off')
        self.estado = estado.text(0,.98,'Preparando…',va='top',fontsize=12,linespacing=1.5,transform=estado.transAxes)
        estado.add_patch(Rectangle((0,.08),1,.055,facecolor='#dce5ed',transform=estado.transAxes))
        self.barra = Rectangle((0,.08),0,.055,facecolor='#0072B2',transform=estado.transAxes)
        estado.add_patch(self.barra)
        self.fig.text(.5,.025,'Amarillo: posición original del bloque. Azul: correspondencia elegida en R. Fondo azulado: parte pendiente.\n'
                      'Se muestra un avance cada medio segundo; el algoritmo evalúa todos los bloques aunque no se dibuje cada candidato.',
                      ha='center',fontsize=10)
        if self.fig.canvas.required_interactive_framework is not None:
            plt.show(block=False)
            self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

    def registrar(self, f, c, df, dc, mae, candidatos, hechos, forzar=False):
        bh,bw = self.C[f:f+16,c:c+16].shape
        bloque = self.R[f+df:f+df+bh,c+dc:c+dc+bw]
        self.prediccion[f:f+bh,c:c+bw] = bloque
        ahora = perf_counter()
        if not plt.fignum_exists(self.fig.number) or (not forzar and hechos != self.total and ahora-self.ultimo < .5):
            return
        self.ultimo = ahora
        self.actual.set_bounds(c-.5,f-.5,bw,bh)
        self.horizontal.set_ydata([f+(bh-1)/2]*2); self.vertical.set_xdata([c+(bw-1)/2]*2)
        f0,f1 = max(0,f-self.p),min(self.R.shape[0],f+bh+self.p)
        c0,c1 = max(0,c-self.p),min(self.R.shape[1],c+bw+self.p)
        self.ventana.set_data(self.R[f0:f1,c0:c1])
        self.ventana.set_extent((c0-.5,c1-.5,f1-.5,f0-.5))
        self.axes[0,1].set_xlim(c0-.5,c1-.5); self.axes[0,1].set_ylim(f1-.5,f0-.5)
        self.sin_desplazar.set_bounds(c-.5,f-.5,bw,bh)
        self.elegido.set_bounds(c+dc-.5,f+df-.5,bw,bh)
        self.formada.set_data(np.ma.masked_invalid(self.prediccion[::self.paso,::self.paso]))
        for im, datos in zip(self.bloques,(self.C[f:f+bh,c:c+bw],bloque)):
            im.set_data(datos); im.set_extent((-.5,bw-.5,bh-.5,-.5))
            im.axes.set_xlim(-.5,bw-.5); im.axes.set_ylim(bh-.5,-.5)
        porcentaje = 100*hechos/self.total
        self.titulo.set_text(f'{self.metodo} · {porcentaje:.1f} % · {hechos}/{self.total} macrobloques')
        self.estado.set_text(f'Último bloque calculado\nFila {f}, columna {c}\n\nVector (df, dc) = ({df}, {dc})\nMAE = {mae:.4f}\nCandidatos: {candidatos}\nTiempo: {ahora-self.inicio:.1f} s')
        self.barra.set_width(hechos/self.total)
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

    def cerrar(self):
        plt.close(self.fig)
