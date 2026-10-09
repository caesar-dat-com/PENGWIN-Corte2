"""Calibración de pérdidas del decoder; detector/backbone congelados, solo train/val."""
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scipy import ndimage as ndi
from torch.nn import functional as F
from torch.utils.data import DataLoader
from entrenar_sesion2 import Samples,collate,seg_loss,decode,evaluate
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.refinamiento_macro import perdida_macro_contorno
from pengwin.metrics_avance3 import gate_semantic

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

@torch.inference_mode()
def edges(model,loader):
    counts=np.zeros((3,4),float)
    for b in loader:
        out=model(b['x']);prob=out['mascaras'].softmax(1).numpy()
        for i in range(len(b['x'])):
            bb,ss,cc=[v.numpy() for v in decode(out,i)]
            pred=gate_semantic(prob[i],bb,cc);gt=b['semantic'][i].numpy()
            for r in (1,2,3):
                pm=pred==r;tm=gt==r;pe=pm&~ndi.binary_erosion(pm);te=tm&~ndi.binary_erosion(tm)
                counts[r-1]+=[np.count_nonzero(pe&ndi.binary_dilation(te,iterations=2)),pe.sum(),np.count_nonzero(te&ndi.binary_dilation(pe,iterations=2)),te.sum()]
    precision=counts[:,0]/np.maximum(counts[:,1],1);recall=counts[:,2]/np.maximum(counts[:,3],1)
    return (2*precision*recall/np.maximum(precision+recall,1e-9)).tolist()

def main():
    torch.set_num_threads(4);out=ROOT/'salidas/cierre/ajuste';out.mkdir(parents=True,exist_ok=True)
    path=ROOT/'salidas/macro_sobel/entrenamiento/best.pth';initial=torch.load(path,map_location='cpu',weights_only=False)
    train,val=Samples(ROOT/'salidas/revision_oct08/datos','train'),Samples(ROOT/'salidas/revision_oct08/datos','val')
    vl=DataLoader(val,batch_size=8,collate_fn=collate)
    variants=[('control',.2,.5,.2),('precision',1.,.25,.5),('contorno',.5,.25,1.)]
    results=[]
    baseline=PelvisSesion2();baseline.load_state_dict(initial['model']);baseline.eval()
    metrics,_=evaluate(baseline,vl,torch.device('cpu'));f=edges(baseline,vl)
    results.append({'name':'inicial','checkpoint':str(path.relative_to(ROOT)),'epoch':0,'metrics':metrics,'boundary_f1':f,'score':.7*metrics['semantic']['macro_dice']+.3*np.mean(f)})
    frozen={k:v.clone() for k,v in initial['model'].items() if not k.startswith('segmentation_head.')}
    for name,bg,mw,cw in variants:
        dest=out/name
        if dest.exists():raise ValueError('No sobrescribir experimento: '+str(dest))
        dest.mkdir();torch.manual_seed(42);np.random.seed(42)
        model=PelvisSesion2();model.load_state_dict(initial['model'])
        for n,p in model.named_parameters():p.requires_grad_(n.startswith('segmentation_head.'))
        opt=torch.optim.AdamW(model.segmentation_head.parameters(),lr=2e-4,weight_decay=1e-4)
        tl=DataLoader(train,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(42),collate_fn=collate)
        config={'status':'running','architecture':'sesion2_skips8_interiores_ds_v1','initial_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'decoder_only':True,'seed':42,'epochs':4,'lr':2e-4,'batch':8,'ce_background':bg,'macro_weight':mw,'contour_weight':cw,'selection':'0.7 Dice semantico + 0.3 F1 contorno tolerancia2px','test_used':False}
        save(dest/'config.json',config);history=[];best=-1
        for epoch in range(1,5):
            model.eval();model.segmentation_head.train();start=time.perf_counter();total=0
            for b in tl:
                opt.zero_grad(set_to_none=True);pred=model(b['x']);target=b['semantic']
                loss,parts=seg_loss(pred,target,b['boundary'],b['cores'])
                ce=F.cross_entropy(pred['mascaras'],target,weight=pred['mascaras'].new_tensor([bg,1,1,1]))
                macro,contour=perdida_macro_contorno(pred['mascaras'],target)
                loss=loss-parts['ce']+ce+mw*macro+cw*contour
                if not torch.isfinite(loss):raise ValueError('Loss no finita')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.segmentation_head.parameters(),5);opt.step();total+=float(loss.detach())*len(b['x'])
            model.eval();metrics,_=evaluate(model,vl,torch.device('cpu'));f=edges(model,vl)
            score=float(.7*metrics['semantic']['macro_dice']+.3*np.mean(f))
            row={'epoch':epoch,'loss':total/len(train),'seconds':time.perf_counter()-start,'metrics':metrics,'boundary_f1':f,'score':score};history.append(row)
            assert all(torch.equal(model.state_dict()[k],v) for k,v in frozen.items()),'Detector/backbone modificados'
            if score>best:
                best=score;torch.save({'model':model.state_dict(),'epoch':epoch,'config':config,'val':metrics},dest/'best.pth');save(dest/'best.json',row)
            save(dest/'historial.json',history)
            print(name,epoch,'Dice',round(metrics['semantic']['macro_dice'],4),'F1 borde',round(float(np.mean(f)),4),flush=True)
        config['status']='completed';save(dest/'config.json',config)
        row=json.loads((dest/'best.json').read_text(encoding='utf-8'));results.append({'name':name,'checkpoint':str((dest/'best.pth').relative_to(ROOT)),**row})
    # Salvaguarda: ninguna anatomía puede empeorar frente al punto de partida.
    reference=results[0]['metrics']['semantic']['dice']
    eligible=[r for r in results if all(x+1e-6>=y for x,y in zip(r['metrics']['semantic']['dice'],reference))]
    selected=max(eligible,key=lambda r:r['score'])
    selected={**selected,'sha256':hashlib.sha256((ROOT/selected['checkpoint']).read_bytes()).hexdigest(),'refinement_strength':0.}
    save(out/'comparacion.json',{'scope':'Calibracion en val, cuatro epocas adicionales iguales por variante; sin test','variants':results,'selected':selected})
    save(ROOT/'salidas/cierre/seleccion_modelo.json',selected);print('SELECCION',selected['name'],flush=True)
if __name__=='__main__':main()
