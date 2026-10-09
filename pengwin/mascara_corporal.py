"""Envolvente corporal obtenida solo del CT, en milímetros y sin etiquetas."""
import numpy as np
from scipy import ndimage as ndi

def mascara_corporal(ct,spacing_zyx,threshold_hu=-500.,opening_mm=2.,margin_mm=3.):
    sp=np.asarray(spacing_zyx,float)
    if ct.ndim!=3 or sp.shape!=(3,) or not np.isfinite(sp).all() or np.any(sp<=0):
        raise ValueError('CT 3D y espaciado positivo requeridos')
    if opening_mm<0 or margin_mm<0:raise ValueError('Radios no negativos requeridos')
    radius=np.ceil(opening_mm/sp[1:]).astype(int)
    y,x=np.ogrid[-radius[0]:radius[0]+1,-radius[1]:radius[1]+1]
    footprint=(y*sp[1])**2+(x*sp[2])**2<=opening_mm**2
    envelope=np.zeros(ct.shape,bool)
    for z in range(len(ct)):
        mask=np.isfinite(ct[z])&(ct[z]>threshold_hu)
        mask=ndi.binary_opening(mask,structure=footprint)
        envelope[z]=ndi.binary_fill_holes(mask)
    labels,n=ndi.label(envelope,ndi.generate_binary_structure(3,3))
    if not n:return envelope
    counts=np.bincount(labels.ravel());counts[0]=0
    body=labels==counts.argmax()
    if margin_mm:body=ndi.distance_transform_edt(~body,sampling=sp)<=margin_mm
    return body
