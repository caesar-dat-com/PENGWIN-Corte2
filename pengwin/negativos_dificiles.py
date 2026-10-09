"""Pérdida sobre fondo anotado que parece hueso o recibe probabilidad pélvica."""
import torch
from torch.nn import functional as F

def perdida_negativos(logits,target,image,fraction=.02):
    if not 0<fraction<=1:raise ValueError('Fracción inválida')
    per_pixel=F.cross_entropy(logits,target,reduction='none')
    foreground=1-logits.softmax(1)[:,0]
    candidate=(target==0)&((image[:,0]>(200+500)/1800)|(foreground.detach()>.5))
    losses=[]
    for loss,mask in zip(per_pixel,candidate):
        values=loss[mask]
        if values.numel():losses.append(values.topk(min(values.numel(),max(1,int(loss.numel()*fraction)))).values.mean())
    return torch.stack(losses).mean() if losses else logits.sum()*0
