"""Entrenamiento sobre pacientes train, selección en val y test reservado."""
import sys,json,argparse,time,hashlib,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from pengwin.models.detector import decodificar_grid
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.fragmentos_sesion2 import objetivos_fragmentos
from pengwin.models.loss import PelvisDetectionLoss
from pengwin.models.nms import nms_por_clase
from pengwin.instances import semantic_labels,fragment_boundaries
from pengwin.metrics_avance3 import detection_metrics,classification_metrics,gate_semantic

def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

class Samples(Dataset):
    def __init__(self,path,split,flip=False):
        self.flip=flip
        splits=json.loads((ROOT/'splits/splits.json').read_text())
        proto=json.loads((path/'protocolo.json').read_text())
        if proto['split_sha256']!=hashlib.sha256((ROOT/'splits/splits.json').read_bytes()).hexdigest():raise ValueError('Splits cambiados')
        self.rows=[r for r in json.loads((path/'manifest.json').read_text()) if r['split']==split]
        if set(r['case'] for r in self.rows)!=set(splits[split]):raise ValueError('Faltan pacientes')
        self.cache={}
        for cid in sorted(set(r['case'] for r in self.rows)):
            # float16 en RAM: con todos los cortes con hueso (~10k) float32 no cabe en Colab.
            with np.load(path/f'{cid}.npz') as a:self.cache[cid]={'images':a['images'].astype(np.float16),'instances':a['instances'].copy(),'boxes':a['boxes'].copy()}
            self.cache[cid]['cores']=np.stack([objetivos_fragmentos(ids)['interiores'] for ids in self.cache[cid]['instances']]).astype(np.float16)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];a=self.cache[r['case']];j=r['index'];ids=a['instances'][j];bb=a['boxes'][j];bb=bb[bb[:,0]>=0]
        img=a['images'][j].astype(np.float32);cores=a['cores'][j].astype(np.float32)
        if self.flip and random.random()<.5:
            # Espejo izquierda-derecha CON intercambio de etiquetas: el coxal que
            # queda del lado izquierdo de la imagen pasa a ser el izquierdo (11..20 <-> 21..30).
            img,cores,ids=img[:,::-1],cores[:,::-1],ids[:,::-1]
            ids=np.where((ids>=11)&(ids<=20),ids+10,np.where(ids>=21,ids-10,ids)).astype(np.uint8)
            bb=bb.copy();bb[:,1],bb[:,3]=1-bb[:,3].copy(),1-bb[:,1].copy();bb[:,0]=np.where(bb[:,0]==1,2,np.where(bb[:,0]==2,1,bb[:,0]))
        ids=np.ascontiguousarray(ids)
        cls=torch.zeros(3)
        for b in bb:cls[int(b[0])]=1
        return {'x':torch.from_numpy(img.copy())[None].repeat(3,1,1),'semantic':torch.from_numpy(semantic_labels(ids).astype(np.int64)),
                'boundary':torch.from_numpy(fragment_boundaries(ids)),'cores':torch.from_numpy(cores.copy()),'boxes':torch.from_numpy(bb.copy()),'classes':cls,'case':r['case'],'z':r['z']}

def collate(samples):
    b={k:torch.stack([s[k] for s in samples]) for k in ('x','semantic','boundary','classes','cores')}
    b['boxes']=torch.cat([torch.cat([torch.full((len(s['boxes']),1),i),s['boxes']],1) for i,s in enumerate(samples)])
    b['meta']=[(s['case'],s['z']) for s in samples];return b

def decode(out,i=0,conf=.25):
    b,s,c=decodificar_grid(out['grid'][i],conf_threshold=conf)
    return nms_por_clase(b,s,c,iou_threshold=.4,score_threshold=conf)

BG_WEIGHT=.2
def seg_loss(out,target,edge,cores):
    # Fondo 0,2 empujaba a pintar hueso: ~81k px de fondo predichos como hueso vs ~170k de hueso real.
    ce=F.cross_entropy(out['mascaras'],target,weight=out['mascaras'].new_tensor([BG_WEIGHT,1.,1.,1.]))
    probs=out['mascaras'].float().softmax(1);gt=F.one_hot(target,4).permute(0,3,1,2).float();dims=(0,2,3)
    dice=1-((2*(probs*gt).sum(dims)+1)/(probs.sum(dims)+gt.sum(dims)+1))[1:].mean()
    raw=F.binary_cross_entropy_with_logits(out['bordes'],edge,reduction='none');positive=edge.bool();negative=(target>0)&~positive
    zero=out['bordes'].sum()*0
    # Equilibrar interfaces raras con interiores anotados, en vez de perderlas en el fondo.
    boundary=(raw[positive].mean() if positive.any() else zero)+(raw[negative].mean() if negative.any() else zero)
    fg=target>0
    raw_core=(out['interiores'].sigmoid()-cores).square()
    interior=(raw_core[fg].mean() if fg.any() else zero)+(raw_core[~fg].mean() if (~fg).any() else zero)
    aux_losses=[]
    for logits in out.get('auxiliares',[]):
        labels=F.interpolate(target[:,None].float(),size=logits.shape[-2:],mode='nearest')[:,0].long()
        aux_losses.append(F.cross_entropy(logits,labels,weight=logits.new_tensor([BG_WEIGHT,1.,1.,1.])))
    aux=torch.stack(aux_losses).mean() if aux_losses else zero
    return ce+1.5*dice+.5*boundary+.5*interior+.3*aux,{'ce':ce,'dice_loss':dice,'boundary':boundary,'interior':interior,'deep_supervision':aux}

GATE_MARGIN=0.
@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval();records=[];truth=[];prob=[];cm=np.zeros((4,4),np.int64)
    for b in loader:
        out=model(b['x'].to(device));sem=out['mascaras'].softmax(1).cpu().numpy()
        truth.extend(b['classes'].tolist());prob.extend(out['clases'].sigmoid().cpu().tolist())
        for i in range(len(b['x'])):
            boxes,scores,labels=[v.cpu().numpy() for v in decode(out,i,.05)]
            records.append({'case':b['meta'][i][0],'z':b['meta'][i][1],'gt':b['boxes'][b['boxes'][:,0]==i,1:].tolist(),'boxes':boxes.tolist(),'scores':scores.tolist(),'labels':labels.tolist()})
            keep=scores>=.25;pred=gate_semantic(sem[i],boxes[keep],labels[keep],GATE_MARGIN);gt=b['semantic'][i].numpy()
            cm+=np.bincount((gt*4+pred).ravel(),minlength=16).reshape(4,4)
    inter=np.diag(cm)[1:];den=cm.sum(1)[1:]+cm.sum(0)[1:]
    dice=2*inter/np.maximum(den,1);iou=inter/np.maximum(den-inter,1)
    return {'detection':detection_metrics(records),'classification':classification_metrics(truth,prob),
            'semantic':{'dice':dice.tolist(),'iou':iou.tolist(),'macro_dice':float(dice.mean()),'confusion':cm.tolist()},
            'scope':'Validación por paciente; Dice semántico no es Dice por fragmento'},records

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=6);p.add_argument('--batch',type=int,default=8);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seg-weight',type=float,default=1.2);p.add_argument('--resume',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'salidas/sesion2/entrenamiento')
    p.add_argument('--data',type=Path,default=ROOT/'salidas/revision_oct08/datos');p.add_argument('--bg-weight',type=float,default=.2)
    p.add_argument('--gate-margin',type=float,default=0.);p.add_argument('--flip',action='store_true');p.add_argument('--canales',type=int,default=8)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--cosine',action='store_true');p.add_argument('--workers',type=int,default=0)
    a=p.parse_args();torch.set_num_threads(a.threads);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    global BG_WEIGHT,GATE_MARGIN;BG_WEIGHT,GATE_MARGIN=a.bg_weight,a.gate_margin
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');data=a.data
    train,val=Samples(data,'train',flip=a.flip),Samples(data,'val');gen=torch.Generator().manual_seed(42)
    tl=DataLoader(train,batch_size=a.batch,shuffle=True,generator=gen,collate_fn=collate,num_workers=a.workers,persistent_workers=a.workers>0)
    vl=DataLoader(val,batch_size=a.batch,collate_fn=collate,num_workers=a.workers)
    model=PelvisSesion2(canales=a.canales).to(device);criterion=PelvisDetectionLoss(balancear_obj=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs*len(tl)) if a.cosine else None
    out=a.output;out.mkdir(parents=True,exist_ok=True);hist=[];best=-1;start=1
    config={'seed':42,'train_patients':70,'val_patients':15,'train_slices':len(train),'val_slices':len(val),'device':str(device),'seg_weight':a.seg_weight,'batch':a.batch,'threads':a.threads,'data_protocol':json.loads((data/'protocolo.json').read_text()),'architecture':'sesion2_skips8_interiores_ds_v1' if a.canales==8 else f'sesion2_skips{a.canales}_interiores_ds_v1','canales':a.canales,'bg_weight':a.bg_weight,'gate_margin':a.gate_margin,'flip_lr_swap':a.flip,'lr':a.lr,'cosine':a.cosine,'status':'running','selection':'media de mAP50 y Dice semántico en val','boundary_threshold':.5}
    if (out/'last.pth').exists():
        if not a.resume:raise ValueError('Salida existente: use --resume o otra --output')
        ck=torch.load(out/'last.pth',map_location=device,weights_only=False)
        for k in ('architecture','seg_weight','batch','data_protocol'):
            if ck['config'][k]!=config[k]:raise ValueError('Reanudación incompatible')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);hist=ck['history'];best=ck['best'];start=ck['epoch']+1
        if sched is not None and ck.get('sched'):sched.load_state_dict(ck['sched'])
        gen.set_state(ck['loader_rng'].cpu());torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type=='cuda' and ck.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    save(out/'config.json',config)
    for epoch in range(start,a.epochs+1):
        model.train();totals={};seen=0;began=time.perf_counter()
        for step,b in enumerate(tl,1):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,enabled=device.type=='cuda'):
                pred=model(b['x'].to(device));ld=criterion(pred,b['boxes'].to(device),b['classes'].to(device))['loss_total']
                ls,terms=seg_loss(pred,b['semantic'].to(device),b['boundary'].to(device),b['cores'].to(device));loss=ld+a.seg_weight*ls
            if not torch.isfinite(loss):raise ValueError('Pérdida no finita')
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(opt);scaler.update()
            if sched is not None:sched.step()
            n=len(b['x']);seen+=n
            for k,v in {'total':loss,'det':ld,'seg':ls,**terms}.items():totals[k]=totals.get(k,0)+float(v.detach())*n
            if step%20==0:print(f'Epoch {epoch}/{a.epochs} batch {step}/{len(tl)} loss={totals["total"]/seen:.4f}',flush=True)
        metrics,records=evaluate(model,vl,device);score=(metrics['detection']['mAP50']+metrics['semantic']['macro_dice'])/2
        row={'epoch':epoch,'seconds':time.perf_counter()-began,'loss':{k:v/seen for k,v in totals.items()},'val':metrics,'selection_score':score};hist.append(row)
        improved=score>best;best=max(best,score)
        ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'history':hist,'best':best,'epoch':epoch,'config':config,'loader_rng':gen.get_state(),'sched':sched.state_dict() if sched is not None else None,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        torch.save(ck,out/'last.tmp.pth');(out/'last.tmp.pth').replace(out/'last.pth')
        if improved:
            torch.save({'model':ck['model'],'config':config,'epoch':epoch,'val':metrics},out/'best.tmp.pth');(out/'best.tmp.pth').replace(out/'best.pth')
            save(out/'metricas_val.json',metrics);save(out/'predicciones_val.json',records)
        save(out/'historial.json',hist);print(f'Epoch {epoch} score={score:.4f} mAP50={metrics["detection"]["mAP50"]:.4f} Dice={metrics["semantic"]["macro_dice"]:.4f}',flush=True)
    config['status']='completed';config['epochs']=len(hist);save(out/'config.json',config)

if __name__=='__main__':main()
