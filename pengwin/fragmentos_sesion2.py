"""Objetivos por fragmento; sus IDs locales no se convierten en clases globales."""
import numpy as np
from scipy import ndimage as ndi
from .instances import semantic_labels, reconstruct_instances


def objetivos_fragmentos(ids):
    if ids.ndim != 2 or not np.isin(ids, np.arange(31)).all():
        raise ValueError('Máscara 2D con IDs enteros 0..30 requerida')
    values = np.unique(ids)
    values = values[values > 0]
    masks = ids[None] == values[:, None, None]
    cores = np.zeros(ids.shape, np.float32)
    for mask in masks:
        # Padding explícito: el fondo exterior existe incluso para objetos
        # que tocan todos los bordes del corte.
        dist = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        cores[mask] = (dist / max(float(dist.max()), 1.))[mask]
    return {'ids': values.astype(np.int64), 'masks': masks,
            'labels': ((values.astype(np.int64)-1)//10+1), 'interiores': cores}


def reconstruir_fragmentos(semantic, boundary, cores, spacing_zyx, **kwargs):
    if cores.shape != semantic.shape:
        raise ValueError('Interiores y semántica deben compartir la cuadrícula')
    # Las semillas requieren interior confiable y ausencia de interfaz.
    # Mantiene la regla de dar una semilla a componentes sin ninguna.
    barrier = np.maximum(boundary, 1.-cores)
    return reconstruct_instances(semantic, barrier, spacing_zyx, **kwargs)
