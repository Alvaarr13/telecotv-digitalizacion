"""Diagrama de los candidatos registrados por LS, sin implementar la búsqueda."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from p8_inter.apoyo import BLOQUES, SALIDA, validar_busqueda


def figura_recorrido_ls(C, R, buscar, p=15, regiones=('vehiculo', 'fondo')):
    """Ejecuta LS con registro y muestra una columna por etapa sobre R.

    buscar conserva sus cinco resultados habituales y acepta traza=None.
    Cada entrada de traza es (d, centro_df, centro_dc, nf, nc, coste).
    Los candidatos se dibujan por el centro de su bloque, en el orden evaluado.
    """
    recorridos = []
    for nombre in regiones:
        f, c = BLOQUES[nombre]
        validar_busqueda(C, R, f, c, p)
        traza = []
        resultado = buscar(C, R, f, c, p, traza=traza)
        if not traza:
            raise ValueError('El diagrama necesita p > 0 y el registro de candidatos dentro de LS.')
        etapas = []
        for d, cf, cc, nf, nc, coste in traza:
            if not etapas or etapas[-1]['d'] != d:
                etapas.append(dict(d=d, centro=(cf, cc), candidatos=[]))
            etapas[-1]['candidatos'].append((nf, nc, float(coste)))
        # min conserva el primer candidato en caso de empate, igual que el contrato LS.
        for etapa in etapas:
            etapa['mejor'] = min(etapa['candidatos'], key=lambda x: x[2])
        ultimo = etapas[-1]['mejor']
        if (tuple(resultado[:2]) != ultimo[:2]
                or not np.isclose(resultado[3], ultimo[2])
                or resultado[4] != len(traza)):
            raise ValueError('El registro no coincide con el resultado de LS; revisar la búsqueda.')
        for anterior, siguiente in zip(etapas, etapas[1:]):
            if anterior['mejor'][:2] != siguiente['centro']:
                raise ValueError('El centro de una etapa no coincide con el ganador anterior.')
        recorridos.append((nombre, f, c, etapas))
    n = max(len(e) for _, _, _, e in recorridos)
    colores = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#6A51A3']
    fig, axes = plt.subplots(len(recorridos), n+1, squeeze=False,
                             figsize=(3.5*(n+1), 4*len(recorridos)+1.2))
    fig.subplots_adjust(left=.045, right=.985, top=.87, bottom=.19, wspace=.3, hspace=.65)
    fig.suptitle(f'Cómo avanza LS · p = {p} píxeles\nCada columna muestra una etapa sobre el fotograma R',
                 fontsize=17, weight='bold', y=.98)
    nombres = {'vehiculo': 'Vehículo en movimiento', 'fondo': 'Fondo fijo', 'pavimento': 'Pavimento'}
    for i, (nombre, f, c, etapas) in enumerate(recorridos):
        bh, bw = C[f:f+16, c:c+16].shape
        # Las coordenadas de puntos y rectángulos se refieren al centro del bloque.
        of, oc = f+(bh-1)/2, c+(bw-1)/2
        contexto = axes[i, 0]
        margen = max(65, p+20)
        f0, f1 = max(0, f-margen), min(C.shape[0], f+bh+margen)
        c0, c1 = max(0, c-margen), min(C.shape[1], c+bw+margen)
        contexto.imshow(C[f0:f1, c0:c1], cmap='gray', vmin=0, vmax=255,
                        extent=(c0-.5,c1-.5,f1-.5,f0-.5))
        contexto.add_patch(Rectangle((c-.5,f-.5), bw,bh, fill=False, ec='#FFD700', lw=2))
        contexto.set_title(nombres.get(nombre, nombre)+'\nBloque actual en C', fontsize=12)
        contexto.set_xlabel(f'Fila {f}, columna {c}', fontsize=10)
        contexto.set_xticks([]); contexto.set_yticks([])
        for j, etapa in enumerate(etapas):
            ax = axes[i,j+1]; color = colores[j % len(colores)]
            m = p+max(bh,bw)/2+3
            rf0, rf1 = max(0,int(of-m)), min(R.shape[0],int(of+m)+1)
            rc0, rc1 = max(0,int(oc-m)), min(R.shape[1],int(oc+m)+1)
            ax.imshow(R[rf0:rf1,rc0:rc1], cmap='gray', vmin=0, vmax=255,
                      extent=(rc0-.5-oc,rc1-.5-oc,rf1-.5-of,rf0-.5-of))
            ax.add_patch(Rectangle((-p,-p),2*p,2*p,fill=False,ec='#555555',ls=':',lw=1.5))
            cf, cc = etapa['centro']; d = etapa['d']
            nf,nc,coste = etapa['mejor']
            # En pasos pequeños se separa el diagrama de puntos de la imagen.
            # Así se conserva la misma escala de contexto en todas las etapas.
            pequeno = d <= 2
            diagrama = ax.inset_axes([.56, .06, .40, .40]) if pequeno else ax
            if pequeno:
                diagrama.set_facecolor('white')
                diagrama.set_title('Detalle de candidatos', fontsize=7, pad=4,
                                   bbox=dict(facecolor='white', edgecolor='none', alpha=.95, pad=1.5))
                radio = 1.65*d
                diagrama.set_xlim(cc-radio, cc+radio)
                diagrama.set_ylim(cf+radio, cf-radio)
                diagrama.set_aspect('equal')
                diagrama.set_xticks([cc-d, cc, cc+d])
                diagrama.set_yticks([cf-d, cf, cf+d])
                diagrama.tick_params(labelsize=6, length=2, pad=1)
                for borde in diagrama.spines.values(): borde.set_edgecolor(color)
            diagrama.add_patch(Rectangle((cc-d,cf-d),2*d,2*d,fill=False,ec=color,ls='--',lw=1))
            for k, (vf,vc,_) in enumerate(etapa['candidatos'],1):
                diagrama.plot(vc,vf,'o',ms=4 if pequeno else 6,mec='white',mew=.6,color=color)
                if d >= 4:
                    diagrama.annotate(str(k),(vc,vf),xytext=(3,4),textcoords='offset points',fontsize=8,color=color,
                                      bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=.3))
            diagrama.plot(cc,cf,marker='+',color='black',ms=6 if pequeno else 9,mew=1.2)
            # Un aro destaca al ganador sin tapar su punto ni los vecinos.
            diagrama.plot(nc,nf,marker='o',ms=12 if pequeno else 15,mfc='none',mec=color,mew=1.8)
            if (nf,nc) != (cf,cc):
                diagrama.annotate('',xy=(nc,nf),xytext=(cc,cf),arrowprops=dict(arrowstyle='->',color=color,lw=1.2))
            ax.add_patch(Rectangle((nc-bw/2,nf-bh/2),bw,bh,fill=False,ec=color,lw=2))
            ax.set_title(f'Etapa {j+1} · paso d = {d}\n{len(etapa["candidatos"])} candidatos evaluados',fontsize=12,color=color)
            ax.set_xlabel(f'Ganador (df, dc) = ({nf}, {nc})\nMAE = {coste:.4f}',fontsize=10)
            ax.set_xlim(-m,m); ax.set_ylim(m,-m); ax.set_aspect('equal')
            ax.set_xticks([-p,0,p]); ax.set_yticks([-p,0,p]); ax.tick_params(labelsize=9)
            if j == 0: ax.set_ylabel('df [píxeles] ↓',fontsize=10)
        for j in range(len(etapas)+1,n+1): axes[i,j].axis('off')
    leyenda = [Line2D([],[],marker='o',ls='',color='#0072B2',label='Centro de cada bloque candidato'),
               Line2D([],[],marker='+',ls='',color='black',label='Centro de la etapa'),
               Line2D([],[],marker='o',ls='',mfc='none',mec='#0072B2',ms=12,label='Ganador de la etapa (aro)'),
               Patch(fill=False,ec='#0072B2',label='Bloque elegido en R'),
               Line2D([],[],ls='--',color='#0072B2',label='Región de la etapa (±d)'),
               Line2D([],[],ls=':',color='#555555',label='Límite de centros permitido (±p)')]
    fig.legend(handles=leyenda,loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,fontsize=10,frameon=False)
    fig.text(.5,.025,'Ejes de las etapas: desplazamientos respecto al bloque de C; dc crece hacia la derecha y df hacia abajo.\n'
             'Con d = 2 y d = 1, los puntos se muestran en un detalle ampliado; el rectángulo sobre R conserva el tamaño real del bloque.',
             ha='center',fontsize=10)
    destino = SALIDA/f'recorrido_LS_p{p}.png'
    fig.savefig(destino,dpi=150)
    plt.show()
    return fig
