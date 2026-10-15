"""Alternativas de instancias: separar solo cuando hay evidencia de interfaz.

No se utiliza GT ni se fuerza un número de fragmentos. Las islas pequeñas se
filtran en unidades físicas y se contabilizan en evaluación, nunca se borra GT.
"""
import numpy as np
from scipy import ndimage as ndi
from .instances import reconstruct_instances
from .fragmentos_sesion2 import reconstruir_fragmentos

def reconstruir(semantic,boundary,cores,spacing,policy):
    method=policy['method'];minimum=policy.get('min_volume_mm3',20.)
    if method=='interiores':
        return reconstruir_fragmentos(semantic,boundary,cores,spacing,min_volume_mm3=minimum,seed_min_volume_mm3=policy.get('seed_mm3',50.))
    if method=='interfaces':
        return reconstruct_instances(semantic,boundary,spacing,threshold=policy['threshold'],min_volume_mm3=minimum,seed_min_volume_mm3=policy.get('seed_mm3',50.),fusionar=policy.get('fusionar',False))
    if method!='componentes':raise ValueError('Método de instancias desconocido')
    sp=np.asarray(spacing)
    if semantic.ndim!=3 or sp.shape!=(3,) or not np.isfinite(sp).all() or (sp<=0).any():raise ValueError('Volumen/spacing inválido')
    pred=np.zeros_like(semantic,dtype=np.int32);mapping={};next_id=1
    for r in (1,2,3):
        labels,_=ndi.label(semantic==r,np.ones((3,3,3),bool));counts=np.bincount(labels.ravel())
        lookup=np.zeros(len(counts),np.int32)
        for i in np.flatnonzero(counts*np.prod(sp)>=minimum):
            if i==0:continue
            lookup[i]=next_id;mapping[next_id]=r;next_id+=1
        pred+=lookup[labels]
    return pred,mapping

def resumen_fragmentos(rows):
    gt=[r for r in rows if r['gt'] is not None]
    return {'gt':len(gt),'pred':sum(r['pred'] is not None for r in rows),'missed':sum(r['pred'] is None for r in rows),
            'extra':sum(r['gt'] is None for r in rows),'matches_iou05':sum(r['gt'] is not None and r['pred'] is not None and r['iou']>=.5 for r in rows),
            'dice_gt':float(np.mean([r['dice'] for r in gt])) if gt else None,'iou_gt':float(np.mean([r['iou'] for r in gt])) if gt else None,
            'dice_symmetric':float(np.mean([r['dice'] for r in rows])) if rows else None}
