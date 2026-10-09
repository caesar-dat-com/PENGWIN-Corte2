"""CLAHE train/inferencia consistente y control de tres épocas ya ejecutado."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from entrenar_sesion2 import Samples,collate,seg_loss
from probar_clahe import clahe_batch,evaluate
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.models.rpn_grid import RPNGridHead
from pengwin.refinamiento_macro import perdida_macro_contorno

def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')

def main():
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    dest=ROOT/'salidas/clahe_revision/adaptacion'
    if dest.exists():raise ValueError('No sobrescribir experimento')
    dest.mkdir(parents=True)
    chosen=json.loads((ROOT/'salidas/cierre/seleccion_modelo.json').read_text(encoding='utf-8'))
    initial=torch.load(ROOT/chosen['checkpoint'],map_location='cpu',weights_only=False)['model']
    model=PelvisSesion2();model.load_state_dict(initial)
    for n,p in model.named_parameters():p.requires_grad_(n.startswith('segmentation_head.'))
    frozen={k:v.clone() for k,v in initial.items() if not k.startswith('segmentation_head.')}
    ck=torch.load(ROOT/'salidas/cajas_revision/rpn_grid/best_diagnostic.pth',map_location='cpu',weights_only=False)
    head=RPNGridHead(ck['config']['anchor_sizes']);head.load_state_dict(ck['head']);head.eval()
    train,val=Samples(ROOT/'salidas/revision_oct08/datos','train'),Samples(ROOT/'salidas/revision_oct08/datos','val')
    tl=DataLoader(train,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(42),collate_fn=collate)
    vl=DataLoader(val,batch_size=8,collate_fn=collate)
    control=PelvisSesion2();control.load_state_dict(torch.load(ROOT/'salidas/negativos_revision/control/best.pth',map_location='cpu',weights_only=False)['model'])
    control_metrics=evaluate(control,head,vl,False);save(dest/'control.json',control_metrics);del control
    config={'scope':'560 real train slices; 120 val; no test','epochs':3,'batch':8,'seed':42,'lr':1e-4,'clip_limit':2,'tile_grid':[8,8],'decoder_only':True,'control':'salidas/negativos_revision/control/best.pth, three epochs same loss/optimizer/seed/order; no extra negative penalty','limitation':'Encoder, detector and RPN are frozen; this is decoder adaptation to CLAHE, not end-to-end detector retraining'}
    save(dest/'config.json',config)
    opt=torch.optim.AdamW(model.segmentation_head.parameters(),lr=1e-4,weight_decay=1e-4)
    history=[];best=-1
    for epoch in range(1,4):
        model.eval();model.segmentation_head.train();start=time.perf_counter();total=0
        for b in tl:
            opt.zero_grad(set_to_none=True);out=model(clahe_batch(b['x']));target=b['semantic']
            loss,parts=seg_loss(out,target,b['boundary'],b['cores'])
            ce=F.cross_entropy(out['mascaras'],target,weight=out['mascaras'].new_tensor([.5,1,1,1]))
            macro,contour=perdida_macro_contorno(out['mascaras'],target)
            loss=loss-parts['ce']+ce+.25*macro+contour
            if not torch.isfinite(loss):raise ValueError('Pérdida no finita')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.segmentation_head.parameters(),5);opt.step();total+=float(loss.detach())*len(target)
        metrics=evaluate(model,head,vl,True)
        assert all(torch.equal(model.state_dict()[k],v) for k,v in frozen.items())
        row={'epoch':epoch,'metrics':metrics,'loss':total/len(train),'seconds':time.perf_counter()-start};history.append(row)
        if metrics['macro_dice']>best:
            best=metrics['macro_dice'];torch.save({'model':model.state_dict(),'config':config,'epoch':epoch,'val':metrics},dest/'best.pth');save(dest/'best.json',row)
        save(dest/'historial.json',history);print(json.dumps(row),flush=True)
    save(dest/'completado.json',{'status':'completed','control':control_metrics,'best':max(history,key=lambda r:r['metrics']['macro_dice'])})

if __name__=='__main__':main()
