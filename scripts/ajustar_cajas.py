"""Ajuste de cajas con GIoU; conserva backbone, máscara y clasificador."""
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from torch.utils.data import DataLoader
from entrenar_sesion2 import Samples,collate,evaluate
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.models.loss import PelvisDetectionLoss
from pengwin.giou import grid_giou_loss

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def main():
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    root=ROOT/'salidas/cierre';selected=json.loads((root/'seleccion_modelo.json').read_text(encoding='utf-8'));initial=ROOT/selected['checkpoint']
    out=root/'ajuste_cajas'
    if out.exists():raise ValueError('No sobrescribir ajuste de cajas')
    out.mkdir();save(out/'seleccion_anterior.json',selected)
    model=PelvisSesion2();model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=False)['model'])
    frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('detection_head.')}
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('detection_head.'))
    train,val=Samples(ROOT/'salidas/revision_oct08/datos','train'),Samples(ROOT/'salidas/revision_oct08/datos','val')
    tl=DataLoader(train,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(42),collate_fn=collate);vl=DataLoader(val,batch_size=8,collate_fn=collate)
    reference,_=evaluate(model,vl,torch.device('cpu'));criterion=PelvisDetectionLoss(balancear_obj=True);opt=torch.optim.AdamW(model.detection_head.parameters(),lr=1e-4,weight_decay=1e-4)
    config={'status':'running','seed':42,'epochs':3,'lr':1e-4,'batch':8,'giou_weight':2.,'initial_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'scope':'Entrenamiento adicional de cabeza detectora y GIoU juntos; no aisla el efecto causal de GIoU','test_used':False};save(out/'config.json',config)
    rows=[];best=reference['detection']['mAP50_95'];accepted=None
    for epoch in range(1,4):
        model.eval();model.detection_head.train();start=time.perf_counter()
        for b in tl:
            opt.zero_grad(set_to_none=True);pred=model(b['x']);loss=criterion(pred,b['boxes'],b['classes'])['loss_total']+2*grid_giou_loss(pred['grid'],b['boxes'])
            if not torch.isfinite(loss):raise ValueError('Loss no finita')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.detection_head.parameters(),5);opt.step()
        model.eval();metrics,_=evaluate(model,vl,torch.device('cpu'))
        assert all(torch.equal(model.state_dict()[k],v) for k,v in frozen.items())
        eligible=metrics['detection']['mAP50']>=reference['detection']['mAP50'] and all(a+1e-6>=b for a,b in zip(metrics['semantic']['dice'],reference['semantic']['dice']))
        row={'epoch':epoch,'metrics':metrics,'seconds':time.perf_counter()-start,'eligible':eligible};rows.append(row)
        if eligible and metrics['detection']['mAP50_95']>best:
            best=metrics['detection']['mAP50_95'];accepted=row;torch.save({'model':model.state_dict(),'config':config,'epoch':epoch,'val':metrics},out/'best.pth')
        save(out/'historial.json',rows);print('CAJAS',epoch,'mAP50',round(metrics['detection']['mAP50'],4),'mAP5095',round(metrics['detection']['mAP50_95'],4),'elegible',eligible,flush=True)
    config['status']='completed';save(out/'config.json',config)
    save(out/'comparacion.json',{'before':reference,'candidates':rows,'accepted':accepted})
    if accepted:
        selected.update(checkpoint=str((out/'best.pth').relative_to(ROOT)),sha256=hashlib.sha256((out/'best.pth').read_bytes()).hexdigest(),metrics=accepted['metrics'],detection_adjustment=accepted['epoch'])
        save(root/'seleccion_modelo.json',selected)
    print('AJUSTE CAJAS FINAL',bool(accepted),flush=True)
if __name__=='__main__':main()
