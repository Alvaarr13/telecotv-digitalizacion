import numpy as np
from p8_inter.apoyo import validar_busqueda, candidato_valido

def estim_mov_fs(C, R, fila, columna, p):
    validar_busqueda(C, R, fila, columna, p)
    actual = C[fila:fila+16, columna:columna+16]
    bh, bw = actual.shape  # Boundary blocks may contain fewer than 16 samples.
    posiciones = [(0, 0)] + [(df, dc) for df in range(-p, p+1)
                            for dc in range(-p, p+1) if (df, dc) != (0, 0)]
    mejor_mae = np.inf
    candidatos = 0
    for df, dc in posiciones:
        if not candidato_valido(R, fila, columna, df, dc, p):
            continue
        bloque = R[fila+df:fila+df+bh, columna+dc:columna+dc+bw]  # Crop bh rows and bw columns starting at (fila+df, columna+dc).
        coste = np.mean(np.abs(actual.astype(np.float64) - bloque.astype(np.float64)))          # Mean absolute difference (actual - bloque): np.mean and np.abs.
        candidatos += 1
        if coste < mejor_mae:  # Retain the first candidate on a tie; zero is evaluated first.
            mejor_mae = coste
            mejor_df, mejor_dc = df, dc  # Retain both offsets of this candidate.
    predicho = R[fila+mejor_df:fila+mejor_df+bh, columna+mejor_dc:columna+mejor_dc+bw]
    error_prediccion =actual.astype(np.float64) - predicho.astype(np.float64)  # Signed difference between actual and predicted blocks.
    return mejor_df, mejor_dc, error_prediccion, mejor_mae, candidatos

def estim_mov_ls(C, R, fila, columna, p, traza=None):
    validar_busqueda(C, R, fila, columna, p)
    actual = C[fila:fila+16, columna:columna+16]
    bh, bw = actual.shape  # Boundary blocks may contain fewer than 16 samples.
    direcciones = [(0,0),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
    df, dc = 0, 0
    candidatos = 0
    d = (p + 1) // 2  # ceil(p/2), using integers.
    if p == 0:
        error_prediccion = actual - R[fila:fila+16, columna:columna+16]
        return 0, 0, error_prediccion, float(np.mean(np.abs(error_prediccion))), 1
    while d >= 1:
        mejor_mae = np.inf
        centro_df, centro_dc = df, dc  # Fixed centre while all nine candidates are considered.
        for paso_f, paso_c in direcciones:
            nf = centro_df + d * paso_f  # Centre row plus d times paso_f.
            nc = centro_dc + d * paso_c  # Centre column plus d times paso_c.
            if not candidato_valido(R, fila, columna, nf, nc, p):
                continue
            bloque = R[fila+nf:fila+nf+bh, columna+nc:columna+nc+bw]
            coste = np.mean(np.abs(actual.astype(np.float64) - bloque.astype(np.float64)))  # Same MAE criterion as FS.
            if traza is not None:
                traza.append((d, centro_df, centro_dc, nf, nc, float(coste)))
            candidatos += 1
            if coste < mejor_mae:
                mejor_mae = coste
                df, dc = nf, nc
        d = d // 2  # Halve the step using integer division.
    predicho = R[fila+df:fila+df+bh, columna+dc:columna+dc+bw]
    return df, dc, actual-predicho, mejor_mae, candidatos

def compensar(R, df, dc):
    from p8_inter.apoyo import validar_par
    validar_par(R, R)
    forma = ((R.shape[0]+15)//16, (R.shape[1]+15)//16)
    if df.shape != forma or dc.shape != forma or df.dtype.kind not in 'iu' or dc.dtype.kind not in 'iu':
        raise ValueError('An integer vector is required for every macroblock, including boundary blocks')
    prediccion = np.empty_like(R)
    for i in range(forma[0]):
        for j in range(forma[1]):
            f, c = 16*i, 16*j
            bh, bw = min(16, R.shape[0]-f), min(16, R.shape[1]-c)
            rf, rc = f + int(df[i,j]), c + int(dc[i,j])
            if not (0 <= rf <= R.shape[0]-bh and 0 <= rc <= R.shape[1]-bw):
                raise ValueError('Motion vector falls outside the reference')
            prediccion[f:f+bh, c:c+bw] = R[rf:rf+bh, rc:rc+bw]  # Crop R: bh rows and bw columns starting at (rf,rc).
    return prediccion