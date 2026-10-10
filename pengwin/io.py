"""Carga de volúmenes PENGWIN (Task 1, CT) y ventaneo en unidades Hounsfield.

Los datos locales están en MetaImage (.mha). Se admiten también NIfTI.
No se deduce su historial de conversión a partir de metadatos heredados.
El espaciado físico (mm/vóxel) se toma SIEMPRE del header vía SimpleITK;
nunca se asume isotrópico ni se reporta nada en píxeles como si fueran mm.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Orientación canónica: todos los casos se reorientan a LPS para que
# "eje x creciente" signifique lo mismo en todos los volúmenes. Sin esto,
# el caso 002 (dirección -1,-1,1) quedaría espejado respecto al 001 y
# un modelo 2D aprendería la lateralidad al revés en la mitad de los cortes.
ORIENTACION = "LPS"

# Taxonomía del dataset (sección 2.1 del enunciado): una región por hueso,
# hasta 10 fragmentos por región. El valor 1/11/21 es el fragmento principal.
REGIONES = {
    1: {"nombre": "Sacro", "sigla": "SA", "rango": (1, 10), "color": "#E4572E"},
    2: {"nombre": "Coxal izquierdo", "sigla": "LI", "rango": (11, 20), "color": "#29335C"},
    3: {"nombre": "Coxal derecho", "sigla": "RI", "rango": (21, 30), "color": "#17BEBB"},
}


def region_de_etiqueta(valor: int) -> int:
    """1..10 -> 1 (sacro), 11..20 -> 2 (coxal izq), 21..30 -> 3 (coxal der), 0 -> 0."""
    if valor == 0:
        return 0
    if not 1 <= valor <= 30:
        raise ValueError(f"Etiqueta fuera de la taxonomía PENGWIN: {valor}")
    return (valor - 1) // 10 + 1


def fragmento_de_etiqueta(valor: int) -> int:
    """Índice del fragmento dentro de su región (1 = principal)."""
    return (valor - 1) % 10 + 1


@dataclass
class Caso:
    id: str
    ct: np.ndarray          # (z, y, x) en HU, float32 (previene desbordamiento de int16 > 32767)
    etiqueta: np.ndarray | None  # (z, y, x) uint8, valores 0..30
    spacing: tuple          # (sz, sy, sx) en mm, mismo orden que el array
    origen: tuple
    direccion_original: tuple


def _leer(ruta: Path) -> sitk.Image:
    img = sitk.ReadImage(str(ruta))
    return normalizar_lps(img)


def normalizar_lps(img: sitk.Image) -> sitk.Image:
    """Reordena ejes sin cambiar puntos físicos, HU o etiquetas.

    No elimina la oblicuidad ni realiza registro anatómico. Rechaza cizalla,
    spacing no positivo y geometría no finita antes de calcular distancias.
    """
    if img.GetDimension()!=3:raise ValueError('Se requiere imagen 3D')
    direction=np.asarray(img.GetDirection()).reshape(3,3)
    if not np.isfinite(direction).all() or not np.isfinite(img.GetOrigin()).all():
        raise ValueError('Geometría no finita')
    if not np.isfinite(img.GetSpacing()).all() or min(img.GetSpacing())<=0:
        raise ValueError('Spacing inválido')
    if not np.allclose(direction.T@direction,np.eye(3),atol=1e-5,rtol=0):
        raise ValueError('Dirección no ortogonal: requiere remuestreo explícito')
    return sitk.DICOMOrient(img,ORIENTACION)


def spacing_zyx(img: sitk.Image) -> tuple:
    """SimpleITK entrega (sx, sy, sz); el array numpy es (z, y, x)."""
    sx, sy, sz = img.GetSpacing()
    return (float(sz), float(sy), float(sx))


def cargar_caso(id_caso: str, dir_imagenes: Path, dir_etiquetas: Path | None = None) -> Caso:
    ruta_ct = resolver_volumen(dir_imagenes, id_caso)
    img = sitk.ReadImage(str(ruta_ct))
    direccion = img.GetDirection()
    img = normalizar_lps(img)
    ct = sitk.GetArrayFromImage(img).astype(np.float32)

    etiqueta = None
    if dir_etiquetas is not None:
        lb = _leer(resolver_volumen(dir_etiquetas, id_caso))
        verificar_alineacion(img, lb, id_caso)
        raw_lb = sitk.GetArrayFromImage(lb)
        unicos = np.unique(raw_lb)
        # Comprobar que todas las etiquetas sean enteras y estén en 0..30
        if not np.all(np.equal(np.mod(unicos, 1), 0)):
            raise ValueError(f"{id_caso}: se encontraron etiquetas no enteras en la máscara: {unicos}")
        if np.any((unicos < 0) | (unicos > 30)):
            raise ValueError(f"{id_caso}: etiquetas fuera del rango oficial PENGWIN 0..30: {unicos}")
        etiqueta = raw_lb.astype(np.uint8)

    return Caso(id_caso, ct, etiqueta, spacing_zyx(img), img.GetOrigin(), direccion)


def verificar_alineacion(img: sitk.Image, lb: sitk.Image, id_caso: str, tol: float = 1e-3) -> None:
    """CT y máscara deben compartir tamaño, espaciado, origen y los 9 componentes de dirección."""
    if img.GetSize() != lb.GetSize():
        raise ValueError(f"{id_caso}: tamaño CT {img.GetSize()} != máscara {lb.GetSize()}")
    for nombre, a, b in [("spacing", img.GetSpacing(), lb.GetSpacing()),
                         ("origen", img.GetOrigin(), lb.GetOrigin()),
                         ("dirección", img.GetDirection(), lb.GetDirection())]:
        if not np.allclose(a, b, atol=tol, rtol=0):
            raise ValueError(f"{id_caso}: {nombre} CT {a} != máscara {b}")


def leer_header(ruta: Path) -> dict:
    """Metadatos sin cargar los vóxeles (rápido, sirve para el QC de los 100 casos)."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(ruta))
    r.ReadImageInformation()
    sx, sy, sz = r.GetSpacing()
    nx, ny, nz = r.GetSize()
    meta = {k: r.GetMetaData(k) for k in r.GetMetaDataKeys()}
    return {
        "nx": nx, "ny": ny, "nz": nz,
        "sx_mm": sx, "sy_mm": sy, "sz_mm": sz,
        "direccion": tuple(round(float(v), 4) for v in r.GetDirection()),  # los 9 componentes completos
        "scl_slope": float(meta.get("scl_slope", 1) or 1),
        "scl_inter": float(meta.get("scl_inter", 0) or 0),
    }


# ---------------------------------------------------------------- ventaneo HU
# Ventanas estándar de radiología (nivel L, ancho W) -> [L - W/2, L + W/2].
VENTANAS = {
    "hueso": (400, 1800),       # [-500, 1300]: cortical y trabecular visibles
    "tejido_blando": (40, 400),
    "hueso_estrecho": (300, 1500),
}


def ventana_hu(ct: np.ndarray, nivel: float = 400, ancho: float = 1800) -> np.ndarray:
    """Recorta a [L-W/2, L+W/2] y normaliza a [0, 1] (float32).

    Es la entrada que verá la red: satura el aire y los metales para que el
    rango dinámico se gaste en hueso, que es lo que se segmenta.
    """
    lo, hi = nivel - ancho / 2, nivel + ancho / 2
    return ((np.clip(ct, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def mascara_hueso_hu(ct: np.ndarray, umbral: float = 200) -> np.ndarray:
    """Umbral clásico de hueso en HU (sin modelo). Base del visualizador 1."""
    return ct >= umbral


def resolver_volumen(directory, case):
    paths = [p for suffix in ('.mha', '.nii', '.nii.gz') for p in Path(directory).rglob(str(case)+suffix)]
    if len(paths) != 1: raise ValueError(f'{case}: se esperaba un volumen, encontrados {len(paths)}')
    return paths[0]


def rutas_dataset():
    import json
    root = Path(__file__).resolve().parents[1]
    cfg = root / 'config_datos.local.json'
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding='utf-8'))
        return Path(data['images']), Path(data['labels'])
    return root / 'data/raw/images', root / 'data/raw/labels'
