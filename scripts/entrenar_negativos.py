"""Comparación controlada de negativos difíciles: decoder propio, train/val."""
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from entrenar_sesion2 import Samples,collate,seg_loss
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.negativos_dificiles import perdida_negativos
from pengwin.refinamiento_macro import perdida_macro_contorno
from pengwin.metrics_avance3 import gate_semantic

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')

@torch.inference_mode()
def evaluate(model,loader,records):
    model.eval();cm=np.zeros((4,4),np.int64);offset=0
    for b in loader:
        prob=model(b['x'])['mascaras'].softmax(1).numpy()
        for i in range(len(prob)):
            row=records[offset];offset+=1
            assert tuple(b['meta'][i])==(row['case'],row['z'])
            pred=gate_semantic(prob[i],np.asarray(row['boxes']).reshape(-1,4),np.asarray(row['labels']))
            gt=b['semantic'][i].numpy();cm+=np.bincount((gt*4+pred).ravel(),minlength=16).reshape(4,4)
    tp=np.diag(cm)[1:];dice=2*tp/np.maximum(cm.sum(0)[1:]+cm.sum(1)[1:],1)
    fg_tp=cm[1:,1:].sum();fp=cm[0,1:].sum();fn=cm[1:,0].sum()
    return {'dice':dice.tolist(),'macro_dice':float(dice.mean()),'false_positive_background_voxels':int(fp),'foreground_recall':float(fg_tp/max(fg_tp+fn,1)),'foreground_precision':float(fg_tp/max(fg_tp+fp,1)),'confusion':cm.tolist()}

def main():
    torch.set_num_threads(4)
    dest=ROOT/'salidas/negativos_revision'
    if dest.exists():raise ValueError('No sobrescribir experimento')
    dest.mkdir()
    chosen=json.loads((ROOT/'salidas/cierre/seleccion_modelo.json').read_text(encoding='utf-8'));path=ROOT/chosen['checkpoint']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==chosen['sha256']
    initial=torch.load(path,map_location='cpu',weights_only=False)['model']
    train,val=Samples(ROOT/'salidas/revision_oct08/datos','train'),Samples(ROOT/'salidas/revision_oct08/datos','val')
    assert set(r['case'] for r in train.rows).isdisjoint(r['case'] for r in val.rows)
    records=json.loads((ROOT/'salidas/cajas_revision/detecciones_fusion.json').read_text(encoding='utf-8'))
    vl=DataLoader(val,batch_size=8,collate_fn=collate)
    model=PelvisSesion2();model.load_state_dict(initial);base=evaluate(model,vl,records)
    save(dest/'baseline.json',base)
    config={'epochs':3,'lr':1e-4,'batch':8,'seed':42,'train_slices':len(train),'val_slices':len(val),'test_used':False,'initial_sha256':chosen['sha256'],'decoder_only':True,'hard_negative_fraction':.02,'variants':{'control':0.,'negativos':.5},'selection':'macro Dice >= baseline; each regional Dice >= baseline-.005; foreground recall >= baseline-.01; fewer background false positives; maximize macro Dice among eligible. Final volume check before deployment.'}
    save(dest/'config.json',config);all_rows=[];frozen={k:v.clone() for k,v in initial.items() if not k.startswith('segmentation_head.')}
    for name,weight in config['variants'].items():
        torch.manual_seed(42);np.random.seed(42);model=PelvisSesion2();model.load_state_dict(initial)
        for n,p in model.named_parameters():p.requires_grad_(n.startswith('segmentation_head.'))
        opt=torch.optim.AdamW(model.segmentation_head.parameters(),lr=config['lr'],weight_decay=1e-4)
        tl=DataLoader(train,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(42),collate_fn=collate)
        folder=dest/name;folder.mkdir();history=[];best=-1;best_eligible=-1
        for epoch in range(1,4):
            model.eval();model.segmentation_head.train();start=time.perf_counter();total=0
            for b in tl:
                opt.zero_grad(set_to_none=True);out=model(b['x']);target=b['semantic']
                loss,parts=seg_loss(out,target,b['boundary'],b['cores'])
                ce=F.cross_entropy(out['mascaras'],target,weight=out['mascaras'].new_tensor([.5,1,1,1]))
                macro,contour=perdida_macro_contorno(out['mascaras'],target)
                loss=loss-parts['ce']+ce+.25*macro+contour+weight*perdida_negativos(out['mascaras'],target,b['x'])
                if not torch.isfinite(loss):raise ValueError('Pérdida no finita')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.segmentation_head.parameters(),5);opt.step();total+=float(loss.detach())*len(target)
            metrics=evaluate(model,vl,records)
            assert all(torch.equal(model.state_dict()[k],v) for k,v in frozen.items())
            eligible=metrics['macro_dice']>=base['macro_dice'] and all(x>=y-.005 for x,y in zip(metrics['dice'],base['dice'])) and metrics['foreground_recall']>=base['foreground_recall']-.01 and metrics['false_positive_background_voxels']<base['false_positive_background_voxels']
            row={'variant':name,'epoch':epoch,'metrics':metrics,'eligible':eligible,'seconds':time.perf_counter()-start,'loss':total/len(train)}
            history.append(row);all_rows.append(row)
            if metrics['macro_dice']>best:
                best=metrics['macro_dice'];torch.save({'model':model.state_dict(),'config':config,'val':metrics,'epoch':epoch},folder/'best.pth')
            if eligible and metrics['macro_dice']>best_eligible:
                best_eligible=metrics['macro_dice'];torch.save({'model':model.state_dict(),'config':config,'val':metrics,'epoch':epoch},folder/'eligible.pth');save(folder/'eligible.json',row)
            save(folder/'historial.json',history);print(json.dumps(row),flush=True)
    eligible=[]
    for name in config['variants']:
        p=dest/name/'eligible.json'
        if p.exists():eligible.append({**json.loads(p.read_text()),'checkpoint':str((dest/name/'eligible.pth').relative_to(ROOT))})
    selected=max(eligible,key=lambda x:x['metrics']['macro_dice']) if eligible else None
    save(dest/'comparacion.json',{'baseline':base,'epochs':all_rows,'selected':selected});print('SELECTED',json.dumps(selected),flush=True)

if __name__=='__main__':main()
