"""Propuestas propias con anclas sobre grid y clasificación anatómica.

Inspirada en RPN: objetidad y deltas de caja por ancla. Se añade clase anatómica
en cada propuesta; no implementa Faster/Mask R-CNN ni usa sus pesos/código.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .nms import nms_por_clase
from ..giou import giou_aligned


class RPNGridHead(nn.Module):
    def __init__(self,anchor_sizes,in_channels=256):
        super().__init__();self.register_buffer('anchor_sizes',torch.as_tensor(anchor_sizes,dtype=torch.float32))
        self.trunk=nn.Sequential(nn.Conv2d(in_channels,128,3,padding=1),nn.GroupNorm(8,128),nn.LeakyReLU(.1),nn.Conv2d(128,64,3,padding=1),nn.GroupNorm(8,64),nn.LeakyReLU(.1))
        self.output=nn.Conv2d(64,len(anchor_sizes)*8,1)
        nn.init.normal_(self.output.weight,std=.01);nn.init.zeros_(self.output.bias)
        with torch.no_grad():self.output.bias.view(-1,8)[:,0]=-2.

    def forward(self,x):return self.output(self.trunk(x))

    def anchors(self,height,width,device):
        yy,xx=torch.meshgrid((torch.arange(height,device=device)+.5)/height,(torch.arange(width,device=device)+.5)/width,indexing='ij')
        center=torch.stack([xx,yy],-1)[None].expand(len(self.anchor_sizes),-1,-1,-1)
        wh=self.anchor_sizes[:,None,None,:].expand(-1,height,width,-1)
        return torch.cat([center-wh/2,center+wh/2],-1).reshape(-1,4)

    def unpack(self,grid):
        b,_,h,w=grid.shape;raw=grid.reshape(b,len(self.anchor_sizes),8,h,w).permute(0,1,3,4,2).reshape(b,-1,8)
        return raw,self.anchors(h,w,grid.device)


def iou_matrix(a,b):
    inter=(torch.minimum(a[:,None,2:],b[None,:,2:])-torch.maximum(a[:,None,:2],b[None,:,:2])).clamp(min=0).prod(-1)
    aa=(a[:,2:]-a[:,:2]).clamp(min=0).prod(-1);bb=(b[:,2:]-b[:,:2]).clamp(min=0).prod(-1)
    return inter/(aa[:,None]+bb[None]-inter).clamp(min=1e-8)


def encode_boxes(boxes,anchors):
    wh=(anchors[:,2:]-anchors[:,:2]).clamp(min=1e-6);center=(anchors[:,:2]+anchors[:,2:])/2
    return torch.cat([((boxes[:,:2]+boxes[:,2:])/2-center)/wh,torch.log((boxes[:,2:]-boxes[:,:2]).clamp(min=1e-6)/wh)],1)


def decode_boxes(deltas,anchors):
    wh=anchors[:,2:]-anchors[:,:2];center=(anchors[:,:2]+anchors[:,2:])/2
    pc=center+deltas[:,:2]*wh;pw=wh*deltas[:,2:].clamp(-4,4).exp()
    return torch.cat([pc-pw/2,pc+pw/2],1)


def targets_for_anchors(anchors,boxes):
    obj=anchors.new_full((len(anchors),),-1.);matched=torch.zeros(len(anchors),dtype=torch.long,device=anchors.device)
    if not len(boxes):obj.zero_();return obj,matched
    overlaps=iou_matrix(anchors,boxes);best,matched=overlaps.max(1)
    obj[best<.3]=0;obj[best>=.5]=1
    # Garantizar una propuesta positiva distinta por GT, incluso en huesos pequeños.
    used=set()
    for j in range(len(boxes)):
        for idx in overlaps[:,j].argsort(descending=True).tolist():
            if idx not in used:used.add(idx);obj[idx]=1;matched[idx]=j;break
    return obj,matched


def proposal_loss(head,grid,targets):
    raw,anchors=head.unpack(grid);terms=[]
    for values,gt in zip(raw,targets):
        obj,match=targets_for_anchors(anchors,gt[:,1:]);positive=torch.where(obj==1)[0];negative=torch.where(obj==0)[0]
        # Negativos difíciles + aleatorios; no diluir positivos entre miles de anclas.
        count=min(len(negative),max(64,len(positive)*3))
        if count:
            hard=negative[values[negative,0].detach().argsort(descending=True)[:count//2]]
            random=negative[torch.randperm(len(negative),device=grid.device)[:count-count//2]]
            neg=torch.unique(torch.cat([hard,random]));loss_obj=F.binary_cross_entropy_with_logits(values[neg,0],obj[neg])
        else:loss_obj=values.sum()*0
        if len(positive):
            loss_obj=loss_obj+F.binary_cross_entropy_with_logits(values[positive,0],obj[positive])
            box_gt=gt[match[positive],1:];enc=encode_boxes(box_gt,anchors[positive]);pr=decode_boxes(values[positive,1:5],anchors[positive])
            reg=F.smooth_l1_loss(values[positive,1:5],enc,beta=1/9)+2*(1-giou_aligned(pr,box_gt)).mean()
            cls=F.cross_entropy(values[positive,5:],gt[match[positive],0].long())
        else:reg=cls=values.sum()*0
        terms.append(loss_obj+reg+cls)
    return torch.stack(terms).mean()


@torch.no_grad()
def decode_proposals(head,grid,index=0,conf=.25,one_per_region=False,presence=None):
    raw,anchors=head.unpack(grid);values=raw[index]
    class_prob,labels=values[:,5:].softmax(1).max(1);scores=values[:,0].sigmoid()*class_prob
    if presence is not None:scores=scores*presence[labels]
    keep=torch.where(scores>=conf)[0]
    if len(keep)>300:keep=keep[scores[keep].argsort(descending=True)[:300]]
    boxes=decode_boxes(values[keep,1:5],anchors[keep]).clamp(0,1);scores=scores[keep];labels=labels[keep]
    valid=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1]);boxes,scores,labels=nms_por_clase(boxes[valid],scores[valid],labels[valid],.4,conf)
    if one_per_region and len(boxes):
        indices=torch.stack([torch.where(labels==c)[0][scores[labels==c].argmax()] for c in torch.unique(labels)])
        boxes,scores,labels=boxes[indices],scores[indices],labels[indices]
    return boxes,scores,labels
