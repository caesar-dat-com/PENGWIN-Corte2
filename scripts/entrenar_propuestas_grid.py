"""Ensayo RPN propio sobre grid, backbone congelado; selección solo en val."""
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from pengwin.models.rpn_grid import RPNGridHead,proposal_loss,decode_proposals
from pengwin.metrics_avance3 import detection_metrics,gate_semantic
from entrenar_sesion2 import Samples

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

@torch.inference_mode()
def evaluate(head,data,truth,one,conf=.25,padding=0.):
    records=[];cm=np.zeros((4,4),np.int64);coverage=[];duplicates=0;missed=0
    head.eval()
    for start in range(0,len(data['features']),8):
        grid=head(data['features'][start:start+8])
        for i in range(len(grid)):
            j=start+i;bb,ss,cc=[v.numpy() for v in decode_proposals(head,grid,i,.05,one,data['classes'][j].sigmoid())]
            bb=np.clip(bb+np.array([-padding,-padding,padding,padding]),0,1);keep=ss>=conf
            records.append({'case':data['meta'][j][0],'z':data['meta'][j][1],'gt':data['boxes'][j].tolist(),'boxes':bb.tolist(),'scores':ss.tolist(),'labels':cc.tolist()})
            pr=gate_semantic(data['probabilities'][j].float().numpy(),bb[keep],cc[keep]);gt=truth[j]
            cm+=np.bincount((gt*4+pr).ravel(),minlength=16).reshape(4,4)
            for label in data['boxes'][j][:,0].long().tolist():
                mask=gt==label+1;inside=np.zeros(mask.shape,bool);boxes=bb[keep&(cc==label)]
                duplicates+=int(len(boxes)>1);missed+=int(len(boxes)==0)
                for b in boxes:
                    x1,y1=np.floor(b[:2]*256).astype(int);x2,y2=np.ceil(b[2:]*256).astype(int);inside[y1:y2,x1:x2]=True
                coverage.append(float((inside&mask).sum()/max(mask.sum(),1)))
    inter=np.diag(cm)[1:];dice=2*inter/np.maximum(cm.sum(0)[1:]+cm.sum(1)[1:],1)
    metrics={'detection':detection_metrics(records),'semantic_dice':dice.tolist(),'macro_dice':float(dice.mean()),'gt_mask_coverage':float(np.mean(coverage)),'duplicate_regions':duplicates,'missed_regions':missed,'one_per_region':one,'conf':conf,'padding':padding}
    return metrics,records

def main():
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    dest=ROOT/'salidas/cajas_revision/rpn_grid'
    if dest.exists():raise ValueError('No sobrescribir ensayo de propuestas')
    dest.mkdir(parents=True);cache=ROOT.parent/'.cajas-cache'
    train=torch.load(cache/'train.pth',weights_only=False);val=torch.load(cache/'val.pth',weights_only=False)
    selected=json.loads((ROOT/'salidas/cierre/seleccion_modelo.json').read_text(encoding='utf-8'));audit=json.loads((ROOT/'salidas/cajas_revision/auditoria.json').read_text(encoding='utf-8'))
    assert train['checkpoint_sha256']==val['checkpoint_sha256']==selected['sha256']
    sizes=[]
    for cls in range(3):
        boxes=torch.cat([b[b[:,0]==cls,1:] for b in train['boxes']]);wh=boxes[:,2:]-boxes[:,:2]
        sizes.extend(torch.quantile(wh,torch.tensor([.25,.5,.75]),dim=0).tolist())
    head=RPNGridHead(sizes);optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4)
    dataset=Samples(ROOT/'salidas/revision_oct08/datos','val');truth=np.stack([dataset[i]['semantic'].numpy() for i in range(len(dataset))]);del dataset
    config={'status':'running','architecture':'RPN propio class-aware sobre grid16, anclas derivadas solo de train, NMS propio','anchor_sizes':sizes,'seed':42,'epochs':12,'batch':8,'lr':3e-4,'backbone_frozen_sha256':selected['sha256'],'loss':'BCE objetidad positivos y negativos difíciles/aleatorios + SmoothL1 deltas beta1/9 + 2GIoU + CE clase','positive_iou':.5,'negative_iou':.3,'forced_positive_per_gt':True,'score':'objetidad * probabilidad de clase por propuesta * presencia anatómica de cabeza propia','train':560,'val':120,'test_used':False,'selection':'Mejor score=.5mAP50:95+.25mAP50+.25coberturaGT; registrar elegibilidad frente al grid anterior antes de reemplazar'}
    save(dest/'config.json',config);history=[];best=-1;eligible_best=-1;eligible_result=None;generator=torch.Generator().manual_seed(42)
    for epoch in range(1,13):
        head.train();order=torch.randperm(len(train['features']),generator=generator);total=0.;start=time.perf_counter()
        for indices in order.split(8):
            optimizer.zero_grad(set_to_none=True);grid=head(train['features'][indices]);loss=proposal_loss(head,grid,[train['boxes'][j] for j in indices.tolist()])
            if not torch.isfinite(loss):raise ValueError('Loss no finita')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5);optimizer.step();total+=float(loss.detach())*len(indices)
        variants=[]
        for one in (False,True):
            metrics,records=evaluate(head,val,truth,one);det=metrics['detection'];score=.5*det['mAP50_95']+.25*det['mAP50']+.25*metrics['gt_mask_coverage']
            eligible=det['mAP50']>=audit['detection']['mAP50'] and det['mean_gt_iou_at_conf025']>=audit['detection']['mean_gt_iou_at_conf025'] and metrics['gt_mask_coverage']>=audit['mean_gt_mask_coverage'] and metrics['macro_dice']>=selected['metrics']['semantic']['macro_dice']-.005
            row={'epoch':epoch,'metrics':metrics,'score':score,'eligible':eligible};variants.append(row)
            if score>best:
                best=score;torch.save({'head':head.state_dict(),'config':config,'val':metrics,'epoch':epoch},dest/'best_diagnostic.pth');save(dest/'best_diagnostic.json',row);save(dest/'detecciones_diagnostic.json',records)
            if eligible and score>eligible_best:
                eligible_best=score;eligible_result=row;torch.save({'head':head.state_dict(),'config':config,'val':metrics,'epoch':epoch},dest/'best_eligible.pth');save(dest/'best_eligible.json',row);save(dest/'detecciones_eligible.json',records)
        history.append({'epoch':epoch,'loss':total/len(train['features']),'seconds':time.perf_counter()-start,'variants':variants});save(dest/'historial.json',history)
        print('RPN GRID',epoch,[(r['metrics']['one_per_region'],round(r['metrics']['detection']['mAP50'],3),round(r['metrics']['detection']['mean_gt_iou_at_conf025'],3),round(r['metrics']['gt_mask_coverage'],3),r['eligible']) for r in variants],flush=True)
    config['status']='completed';save(dest/'config.json',config);save(dest/'seleccion.json',{'eligible':eligible_result,'baseline':audit,'note':'Ensayo de validación; no sustituye ni reevalúa automáticamente el protocolo test anterior.'})
    print('RPN GRID TERMINADO',eligible_result,flush=True)
if __name__=='__main__':main()
