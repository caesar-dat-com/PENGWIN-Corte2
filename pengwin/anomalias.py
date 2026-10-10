"""Control de anomalías de la selección de huesos (entrada CT y salida del modelo).

Dos niveles:
- Entrada: el CT trae geometría o HU que el modelo nunca vio (cizalla, cortes
  muy gruesos, metal, rango HU imposible). Solo se marca: no se corrige el CT.
- Salida: la máscara semántica predicha (z, y, x) con 0 fondo, 1 sacro,
  2 coxal izquierdo, 3 coxal derecho, en orientación LPS. Se marcan y, si se
  pide, se corrigen errores anatómicamente imposibles: lateralidad invertida,
  hueso en aire, islas de uno o dos cortes sin continuidad en z.

Ninguna regla usa GT. Las referencias de volumen se ajustan con train
(`referencia_desde_gt`) y se aplican a val/test sin reajustar.
"""
import numpy as np
from scipy import ndimage as ndi

NOMBRES = {1: 'sacro', 2: 'coxal_izq', 3: 'coxal_der'}


def _anomalia(tipo, severidad, detalle, region=None, **extra):
    return {'tipo': tipo, 'severidad': severidad, 'region': NOMBRES.get(region), 'detalle': detalle, **extra}


def anomalias_entrada(image, hu=None):
    """Chequeos de un sitk.Image ya reorientado a LPS (y su array HU opcional)."""
    rows = []
    sx, sy, sz = image.GetSpacing(); nx, ny, nz = image.GetSize()
    d = np.asarray(image.GetDirection()).reshape(3, 3)
    if not np.allclose(d.T @ d, np.eye(3), atol=1e-4):
        rows.append(_anomalia('geometria_cizalla', 'alta', 'Ejes no ortogonales: EDT y distancias en mm no son válidas'))
    if not np.allclose(np.abs(d), np.eye(3), atol=1e-3):
        rows.append(_anomalia('geometria_oblicua', 'media', 'Dirección oblicua tras LPS; lateralidad por eje x aproximada'))
    if sz > 5:
        rows.append(_anomalia('cortes_gruesos', 'media', f'Espaciado z {sz:.2f} mm > 5 mm: continuidad en z poco fiable'))
    if nz < 40:
        rows.append(_anomalia('pocos_cortes', 'media', f'Solo {nz} cortes: la pelvis puede estar incompleta'))
    if max(sx, sy) > 1.5:
        rows.append(_anomalia('resolucion_baja', 'baja', f'Píxel {sx:.2f}×{sy:.2f} mm: fragmentos finos pueden perderse'))
    if hu is not None:
        lo, hi = float(np.percentile(hu, 0.5)), float(np.percentile(hu, 99.9))
        if lo > -500 or hi < 300:
            rows.append(_anomalia('rango_hu', 'alta', f'Percentiles HU [{lo:.0f}, {hi:.0f}]: no parece CT en HU (¿slope/intercept?)'))
        metal = float((hu > 3000).mean())
        if metal > 1e-4:
            rows.append(_anomalia('metal', 'media', f'{metal:.4%} de vóxeles > 3000 HU: artefacto metálico probable', fraccion=metal))
    return rows


def _linea_media(hu, sem):
    """x de la línea media por corte: centroide del cuerpo (HU > -300); sacro si existe."""
    nz, ny, nx = sem.shape; xs = np.arange(nx)
    media = np.full(nz, nx / 2, float)
    for z in range(nz):
        sac = sem[z] == 1
        if sac.sum() > 30:
            media[z] = (sac.sum(0) * xs).sum() / sac.sum(); continue
        if hu is not None:
            body = hu[z] > -300
            if body.sum() > 100:
                media[z] = (body.sum(0) * xs).sum() / body.sum()
    # Suavizar: la línea media no salta entre cortes contiguos.
    return ndi.median_filter(media, size=9, mode='nearest')


def corregir_lateralidad(sem, hu=None):
    """Cada componente 2D de coxal se reasigna al lado de la línea media donde está.

    En LPS el eje x crece hacia la izquierda del paciente: coxal izquierdo
    debe quedar con x mayor que la línea media. Se decide por componente (no
    por píxel) para no partir un hueso que cruza la sínfisis.
    """
    out = sem.copy(); media = _linea_media(hu, sem); cambiados = 0
    for z in range(sem.shape[0]):
        cox = (sem[z] == 2) | (sem[z] == 3)
        if not cox.any(): continue
        lab, n = ndi.label(cox)
        for i, (sl_y, sl_x) in enumerate(ndi.find_objects(lab), 1):
            comp = lab[sl_y, sl_x] == i
            cx = sl_x.start + (comp.sum(0) * np.arange(comp.shape[1])).sum() / comp.sum()
            lado = 2 if cx > media[z] else 3
            region = out[z, sl_y, sl_x]
            mal = comp & (region != lado)
            if mal.any():
                cambiados += int(mal.sum()); region[comp] = lado
    return out, cambiados


def quitar_aire(sem, hu, umbral_hu=-500):
    """Hueso predicho sobre aire (GT real llega a -400 HU en médula grasa): imposible, se pasa a fondo."""
    malo = (sem > 0) & (hu < umbral_hu)
    out = sem.copy(); out[malo] = 0
    return out, int(malo.sum())


def quitar_islas_z(sem, min_cortes=3):
    """Componentes 3D de una región que ocupan menos de `min_cortes` cortes y
    no son la componente mayor de esa región: ruido corte a corte del modelo 2D."""
    out = sem.copy(); quitados = 0
    for r in (1, 2, 3):
        lab, n = ndi.label(sem == r, np.ones((3, 3, 3), bool))
        if n <= 1: continue
        tam = np.bincount(lab.ravel()); tam[0] = 0; mayor = int(tam.argmax())
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            if i == mayor: continue
            if sl[0].stop - sl[0].start < min_cortes:
                m = lab[sl] == i; out[sl][m] = 0; quitados += int(m.sum())
    return out, quitados


def suavizar_z(sem):
    """Voto de mayoría en ventana de 3 cortes por píxel (solo cambia si 2 de 3 coinciden)."""
    out = sem.copy()
    a, b, c = sem[:-2], sem[1:-1], sem[2:]
    voto = np.where(a == c, a, b)
    out[1:-1] = voto
    return out, int((out != sem).sum())


def referencia_desde_gt(volumenes):
    """volumenes: lista de dicts {region: mm3} medidos en GT de train."""
    ref = {}
    for r in (1, 2, 3):
        v = np.array([x[r] for x in volumenes if x.get(r)], float)
        if len(v): ref[r] = {'p02': float(np.percentile(v, 2)), 'p98': float(np.percentile(v, 98)), 'n': int(len(v))}
    return ref


def volumen_regiones(sem, spacing_zyx):
    vox = float(np.prod(spacing_zyx)); cuenta = np.bincount(sem.ravel(), minlength=4)
    return {r: float(cuenta[r] * vox) for r in (1, 2, 3)}


def controlar(sem, spacing_zyx, hu=None, referencia=None, instancias=None, mapping=None, corregir=True):
    """Devuelve (semantica_corregida, informe). `hu` debe estar en la misma grilla que `sem`."""
    if sem.ndim != 3: raise ValueError('Se requiere volumen 3D contiguo')
    rows = []; acciones = {}; out = sem.astype(np.uint8)
    if hu is not None:
        aire = int(((out > 0) & (hu < -500)).sum())
        if aire:
            rows.append(_anomalia('hueso_en_aire', 'alta', f'{aire} vóxeles de hueso con HU < -500 (aire)'))
            if corregir: out, acciones['hueso_en_aire'] = quitar_aire(out, hu)
    lat, n_lat = corregir_lateralidad(out, hu)
    if n_lat:
        rows.append(_anomalia('lateralidad', 'alta', f'{n_lat} vóxeles de coxal del lado contrario a la línea media'))
        if corregir: out = lat; acciones['lateralidad'] = n_lat
    isl, n_isl = quitar_islas_z(out)
    if n_isl:
        rows.append(_anomalia('islas_z', 'media', f'{n_isl} vóxeles en componentes de < 3 cortes'))
        if corregir: out = isl; acciones['islas_z'] = n_isl
    if corregir:
        out, acciones['suavizado_z'] = suavizar_z(out)
    vols = volumen_regiones(out, spacing_zyx)
    for r in (1, 2, 3):
        if vols[r] == 0:
            rows.append(_anomalia('region_ausente', 'alta', 'Región sin vóxeles en todo el volumen', r)); continue
        if referencia and r in referencia:
            lo, hi = referencia[r]['p02'], referencia[r]['p98']
            if not lo * .5 <= vols[r] <= hi * 1.5:
                rows.append(_anomalia('volumen_fuera_de_rango', 'media', f'{vols[r]/1000:.0f} cm³ fuera de [{lo*.5/1000:.0f}, {hi*1.5/1000:.0f}] (train)', r, volumen_mm3=vols[r]))
        if hu is not None:
            m = float(hu[out == r].mean())
            if m < 150:
                rows.append(_anomalia('hueso_hipodenso', 'media', f'HU medio {m:.0f} < 150: probable tejido blando', r))
    if instancias is not None and mapping is not None:
        presentes = set(np.unique(instancias)) - {0}
        for r in (1, 2, 3):
            n = sum(1 for i, c in mapping.items() if c == r and i in presentes)
            if n > 10:
                rows.append(_anomalia('fragmentos_excesivos', 'alta', f'{n} instancias; la taxonomía admite máx. 10', r, n=n))
    sev = {'alta': 3, 'media': 2, 'baja': 1}
    puntaje = sum(sev[a['severidad']] for a in rows)
    return out, {'anomalias': rows, 'acciones': acciones, 'puntaje': puntaje,
                 'estado': 'revisar' if any(a['severidad'] == 'alta' for a in rows) else 'ok',
                 'volumenes_mm3': vols}
