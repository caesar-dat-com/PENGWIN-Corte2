"""GIoU propio para cajas normalizadas; no requiere detectores externos."""
import torch

def giou_aligned(pred,target):
    low=torch.maximum(pred[:,:2],target[:,:2]);high=torch.minimum(pred[:,2:],target[:,2:])
    inter=(high-low).clamp(min=0).prod(1)
    pa=(pred[:,2:]-pred[:,:2]).clamp(min=0).prod(1);ta=(target[:,2:]-target[:,:2]).clamp(min=0).prod(1)
    union=pa+ta-inter
    enclosing=(torch.maximum(pred[:,2:],target[:,2:])-torch.minimum(pred[:,:2],target[:,:2])).clamp(min=0).prod(1)
    return inter/union.clamp(min=1e-8)-(enclosing-union)/enclosing.clamp(min=1e-8)

def grid_giou_loss(grid,boxes):
    if not len(boxes):return grid.sum()*0
    height,width=grid.shape[-2:];batch=boxes[:,0].long();gt=boxes[:,2:6]
    center=(gt[:,:2]+gt[:,2:])/2
    ix=(center[:,0]*width).long().clamp(0,width-1);iy=(center[:,1]*height).long().clamp(0,height-1)
    raw=grid[batch,1:5,iy,ix].sigmoid()
    cx=(raw[:,0]+ix)/width;cy=(raw[:,1]+iy)/height;w,h=raw[:,2],raw[:,3]
    pred=torch.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],1)
    return (1-giou_aligned(pred,gt)).mean()
