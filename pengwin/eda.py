"""Análisis exploratorio por caso: fragmentos, volúmenes y separación en mm.

Se calcula solo con las máscaras de referencia (ground truth), así que corre
sobre los 100 casos sin necesitar las imágenes CT.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from .io import ORIENTACION, REGIONES, fragmento_de_etiqueta, region_de_etiqueta, spacing_zyx


def _bbox(mask: np.ndarray) -> tuple:
    idx = np.nonzero(mask)
    return tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)


def _union_bbox(a: tuple, b: tuple, forma: tuple, margen: int = 2) -> tuple:
    return tuple(slice(max(0, min(x.start, y.start) - margen), min(n, max(x.stop, y.stop) + margen))
                 for x, y, n in zip(a, b, forma))


def distancia_borde_a_borde(principal: np.ndarray, fragmento: np.ndarray, spacing: tuple) -> float:
    """Distancia mínima (mm) entre la superficie del fragmento y la del principal.

    distance_transform_edt con `sampling=spacing` mide en mm reales. El EDT
    de ~principal da, en cada vóxel, la distancia al vóxel del principal más
    cercano (centro a centro); el mínimo sobre el fragmento es la separación.
    Si se tocan, el resultado es del orden de un vóxel (~0.8 mm), no 0.
    """
    caja = _union_bbox(_bbox(principal), _bbox(fragmento), principal.shape)
    edt = ndimage.distance_transform_edt(~principal[caja], sampling=spacing)
    return float(edt[fragmento[caja]].min())


def resumen_caso(ruta_etiqueta: Path) -> tuple[dict, list[dict]]:
    """Devuelve (fila por caso, filas por fragmento)."""
    id_caso = Path(ruta_etiqueta).stem
    img = sitk.DICOMOrient(sitk.ReadImage(str(ruta_etiqueta)), ORIENTACION)
    lb = sitk.GetArrayFromImage(img)
    sp = spacing_zyx(img)
    ml_por_voxel = float(np.prod(sp)) / 1000.0

    valores, cuentas = np.unique(lb, return_counts=True)
    vol = {int(v): int(c) for v, c in zip(valores, cuentas) if v != 0}

    caso = {"caso": id_caso, "nz": lb.shape[0], "ny": lb.shape[1], "nx": lb.shape[2],
            "sz_mm": sp[0], "sy_mm": sp[1], "sx_mm": sp[2]}
    fragmentos = []

    # Cortes axiales que contienen cada región (define el desbalance 2D)
    region_vol = np.zeros_like(lb, dtype=np.uint8)
    for v in vol:
        region_vol[lb == v] = region_de_etiqueta(v)
    for r, info in REGIONES.items():
        caso[f"cortes_{info['sigla']}"] = int((region_vol == r).any(axis=(1, 2)).sum())
    caso["cortes_con_hueso"] = int((region_vol > 0).any(axis=(1, 2)).sum())

    for r, info in REGIONES.items():
        lo, hi = info["rango"]
        etiquetas = sorted(v for v in vol if lo <= v <= hi)
        s = info["sigla"]
        caso[f"n_frag_{s}"] = len(etiquetas)
        caso[f"vol_ml_{s}"] = round(sum(vol[v] for v in etiquetas) * ml_por_voxel, 2)
        if not etiquetas:
            continue
        mayor = max(etiquetas, key=vol.get)
        caso[f"principal_es_mayor_{s}"] = mayor == lo
        principal = lb == lo if lo in vol else lb == mayor

        for v in etiquetas:
            frag = {"caso": id_caso, "region": s, "etiqueta": v,
                    "fragmento": fragmento_de_etiqueta(v),
                    "vol_ml": round(vol[v] * ml_por_voxel, 3),
                    "es_principal": v == lo}
            if v != lo:
                m = lb == v
                frag["n_componentes"] = int(ndimage.label(m)[1])
                d = distancia_borde_a_borde(principal, m, sp)
                frag["dist_mm_gt"] = round(d, 2)
                # vecindad-26: la diagonal de un vóxel es el máximo para "tocarse"
                frag["en_contacto"] = d <= float(np.sqrt(np.sum(np.square(sp)))) + 1e-6
            fragmentos.append(frag)

    caso["n_frag_total"] = sum(caso[f"n_frag_{i['sigla']}"] for i in REGIONES.values())
    caso["regiones_fracturadas"] = sum(caso[f"n_frag_{i['sigla']}"] > 1 for i in REGIONES.values())
    return caso, fragmentos


def estadisticas_hu(ruta_ct: Path, ruta_etiqueta: Path, max_muestras: int = 200_000,
                    seed: int = 42) -> dict:
    """Percentiles de HU dentro del hueso anotado vs. el volumen completo.

    Justifica la ventana y el umbral: si el p5 del hueso anotado está por
    debajo de 200 HU, el umbral clásico pierde trabecular (sacro, sobre todo).
    """
    ct = sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(str(ruta_ct)), ORIENTACION))
    lb = sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(str(ruta_etiqueta)), ORIENTACION))
    rng = np.random.default_rng(seed)
    hueso = ct[lb > 0]
    if hueso.size > max_muestras:
        hueso = rng.choice(hueso, max_muestras, replace=False)
    fila = {"caso": Path(ruta_ct).stem, "hu_min": int(ct.min()), "hu_max": int(ct.max())}
    for p in (1, 5, 25, 50, 75, 95, 99):
        fila[f"hueso_p{p}"] = float(np.percentile(hueso, p))
    fila["frac_hueso_bajo_200HU"] = float((hueso < 200).mean())
    return fila
