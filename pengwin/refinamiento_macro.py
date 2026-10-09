"""Máscara macro, contorno Sobel y refinamiento local sin consultar GT."""
import numpy as np
import torch
from torch.nn import functional as F
from scipy import ndimage as ndi


def sobel_tensor(x):
    """Gradiente por canal, normalizado /8; padding replicado, diferenciable."""
    k=x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/8
    n=x.shape[1]
    pad=F.pad(x,(1,1,1,1),mode='replicate')
    gx=F.conv2d(pad,k.expand(n,1,3,3),groups=n)
    gy=F.conv2d(pad,k.T.expand(n,1,3,3),groups=n)
    return gx,gy


def perdida_macro_contorno(logits,target):
    """Supervisa la unión de huesos y contornos anatómicos, no IDs de fragmentos."""
    prob=logits.float().softmax(1)
    bone=1-prob[:,0];gt=(target>0).float()
    raw=F.binary_cross_entropy(bone.clamp(1e-6,1-1e-6),gt,reduction='none')
    zero=logits.sum()*0
    bce=sum(raw[m].mean() if m.any() else zero for m in (gt.bool(),~gt.bool()))
    dims=(1,2)
    dice=1-((2*(bone*gt).sum(dims)+1)/(bone.sum(dims)+gt.sum(dims)+1)).mean()
    truth=F.one_hot(target,4).permute(0,3,1,2).float()[:,1:]
    px,py=sobel_tensor(prob[:,1:]);tx,ty=sobel_tensor(truth)
    error=(px-tx).abs()+(py-ty).abs()
    band=F.max_pool2d(((tx.abs()+ty.abs())>0).float(),3,1,1).bool()
    # Equilibrar la banda de contorno y su exterior: el fondo no domina.
    contour=sum(error[m].mean() if m.any() else zero for m in (band,~band))
    return bce+dice,contour


def refinar_probabilidades(prob,image,strength=.25,iterations=1,sigma=.10,confidence=.80):
    """Vecindad 8 (corte 2D), bilateral 3x3, solo píxeles inciertos.

    No cierre morfológico, eliminación por tamaño ni selección del mayor hueso.
    strength=0 es identidad. sigma está en intensidad ventaneada [0,1].
    """
    p=np.asarray(prob,dtype=np.float32)
    im=np.asarray(image,dtype=np.float32)
    if p.ndim!=3 or p.shape[0]!=4 or p.shape[1:]!=im.shape:raise ValueError('Cuadrículas incompatibles')
    if not np.isfinite(p).all() or not np.isfinite(im).all() or (p<0).any():raise ValueError('Valores inválidos')
    if not np.allclose(p.sum(0),1,atol=1e-4):raise ValueError('Se requieren probabilidades normalizadas')
    if not 0<=strength<=1 or sigma<=0 or iterations<0:raise ValueError('Parámetros inválidos')
    if strength==0:return p.copy()
    original=p.copy();h,w=im.shape;ip=np.pad(im,1,mode='edge')
    uncertain=original.max(0)<confidence
    for _ in range(iterations):
        padded=np.pad(p,((0,0),(1,1),(1,1)),mode='edge')
        total=p.copy();den=np.ones(im.shape,np.float32)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if not (dx or dy):continue
                neighbor=ip[1+dy:1+dy+h,1+dx:1+dx+w]
                weight=np.exp(-((im-neighbor)/sigma)**2/2)/(dx*dx+dy*dy)
                total+=padded[:,1+dy:1+dy+h,1+dx:1+dx+w]*weight
                den+=weight
        proposal=(1-strength)*original+strength*total/den
        p=np.where(uncertain[None],proposal,original)
        p/=p.sum(0,keepdims=True)
    return p


def control_calidad(semantic,prob=None):
    """Alertas heurísticas de segmentación; no detector clínico de anomalías."""
    sem=np.asarray(semantic)
    if sem.ndim not in (2,3) or not np.isin(sem,[0,1,2,3]).all():raise ValueError('Semántica inválida')
    structure=ndi.generate_binary_structure(sem.ndim,sem.ndim)
    regions=[];alerts=[]
    for r in (1,2,3):
        labels,n=ndi.label(sem==r,structure)
        sizes=np.bincount(labels.ravel())[1:]
        regions.append({'region':r,'components':int(n),'voxels':int(sizes.sum()),'largest_fraction':float(sizes.max()/sizes.sum()) if n else None})
        # El límite de 10 IDs por hueso se aplica al volumen, no a cortes 2D.
        if sem.ndim==3 and n>10:alerts.append(f'region_{r}: mas de 10 componentes; revisar ruido, desconexiones y fracturas')
    result={'kind':'control de calidad, no diagnostico','connectivity':26 if sem.ndim==3 else 8,'regions':regions,'alerts':alerts}
    if prob is not None:
        p=np.asarray(prob);fg=sem>0
        entropy=-(p*np.log(np.clip(p,1e-8,1))).sum(0)/np.log(p.shape[0])
        result['mean_foreground_entropy']=float(entropy[fg].mean()) if fg.any() else None
    return result
