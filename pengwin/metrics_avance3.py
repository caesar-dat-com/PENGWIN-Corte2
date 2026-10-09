"""Métricas explícitas: AP interpolada 101 puntos, AUC por rangos, Dice semántico."""
import numpy as np
from scipy.stats import rankdata

def box_iou(a,b):
    b=np.asarray(b).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[2:],b[:,2:])-np.maximum(a[:2],b[:,:2])).prod(1)
    return inter/np.maximum(np.prod(a[2:]-a[:2])+np.prod(b[:,2:]-b[:,:2],axis=1)-inter,1e-9)

def detection_metrics(records):
    aps=[]; details={}; gt_ious=[]
    for c in range(3):
        gt={i:np.asarray(r['gt']).reshape(-1,5)[np.asarray(r['gt']).reshape(-1,5)[:,0]==c,1:] for i,r in enumerate(records)}
        detections=[(float(s),i,np.asarray(b)) for i,r in enumerate(records) for b,s,k in zip(r['boxes'],r['scores'],r['labels']) if k==c]
        detections.sort(key=lambda t:-t[0]); ng=sum(map(len,gt.values())); values=[]
        for threshold in np.arange(.5,.96,.05):
            used={i:set() for i in gt};tp=[];fp=[]
            for score,i,b in detections:
                overlaps=box_iou(b,gt[i]); order=np.argsort(-overlaps)
                match=next((int(j) for j in order if j not in used[i] and overlaps[j]>=threshold-1e-8),None)
                tp.append(match is not None);fp.append(match is None)
                if match is not None:used[i].add(match)
            t=np.cumsum(tp);f=np.cumsum(fp);rec=t/max(ng,1);prec=t/np.maximum(t+f,1)
            ap=float(np.mean([prec[rec>=r].max() if np.any(rec>=r) else 0. for r in np.linspace(0,1,101)])) if ng else None
            values.append(ap)
        details[str(c)]={'gt':ng,'ap50':values[0],'ap50_95':float(np.mean(values)) if ng else None}
        if ng:aps.append(values)
        for i,gts in gt.items():
            pb=[b for score,j,b in detections if i==j and score>=.25]
            gt_ious.extend(float(box_iou(g,pb).max()) if pb else 0. for g in gts)
    return {'mAP50':float(np.mean([v[0] for v in aps])) if aps else None,
            'mAP50_95':float(np.mean(aps)) if aps else None,'per_class':details,
            'mean_gt_iou_at_conf025':float(np.mean(gt_ious)) if gt_ious else None,
            'definition':'AP 101 puntos, IoU 0.50:0.05:0.95, NMS propio, sin filtros COCO por área'}

def classification_metrics(truth,prob):
    y=np.asarray(truth,bool);p=np.asarray(prob);pred=p>=.5;rows=[]
    for c in range(3):
        tp=int((y[:,c]&pred[:,c]).sum());fp=int((~y[:,c]&pred[:,c]).sum());fn=int((y[:,c]&~pred[:,c]).sum());tn=int((~y[:,c]&~pred[:,c]).sum())
        pos=int(y[:,c].sum());neg=len(y)-pos
        auc=float((rankdata(p[:,c])[y[:,c]].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None
        rows.append({'class':c,'f1':2*tp/max(2*tp+fp+fn,1),'auc':auc,'tp':tp,'fp':fp,'fn':fn,'tn':tn})
    aucs=[r['auc'] for r in rows if r['auc'] is not None]
    return {'per_class':rows,'macro_f1':float(np.mean([r['f1'] for r in rows])),'macro_auc':float(np.mean(aucs)) if aucs else None}

def gate_semantic(prob,boxes,labels):
    """Cada región solo puede ocupar píxeles dentro de su bbox predicha tras NMS."""
    h,w=prob.shape[-2:];allowed=np.zeros((4,h,w),bool);allowed[0]=True
    for box,c in zip(boxes,labels):
        x1,y1,x2,y2=box
        left,top=max(0,int(np.floor(x1*w))),max(0,int(np.floor(y1*h)))
        right,bottom=min(w,int(np.ceil(x2*w))),min(h,int(np.ceil(y2*h)))
        allowed[int(c)+1,top:bottom,left:right]=True
    return np.where(allowed,prob,-1).argmax(0).astype(np.uint8)
