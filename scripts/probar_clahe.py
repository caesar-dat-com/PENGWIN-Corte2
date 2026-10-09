"""Ensayo CLAHE reproducible con detección recalculada; solo validación."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np,torch
from torch.utils.data import DataLoader
from entrenar_sesion2 import Samples,collate,decode
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.models.rpn_grid import RPNGridHead,decode_proposals
from pengwin.propuestas_regionales import combinar_regiones
from pengwin.metrics_avance3 import gate_semantic,detection_metrics
from pengwin.contraste import clahe_batch

@torch.inference_mode()
def evaluate(model,head,loader,use_clahe):
    model.eval();records=[];cm=np.zeros((4,4),np.int64)
    for b in loader:
        x=clahe_batch(b['x']) if use_clahe else b['x'];out=model(x);proposals=head(out['features'])
        for i in range(len(x)):
            ob,os,oc=[v.numpy() for v in decode(out,i,.05)]
            nb,ns,nc=[v.numpy() for v in decode_proposals(head,proposals,i,.05,presence=out['clases'][i].sigmoid())]
            bb,ss,cc,_=combinar_regiones(ob,os,oc,nb,ns,nc,new_conf=.1,old_conf=.25,padding=0)
            pred=gate_semantic(out['mascaras'][i].softmax(0).numpy(),bb,cc);gt=b['semantic'][i].numpy()
            cm+=np.bincount((gt*4+pred).ravel(),minlength=16).reshape(4,4)
            records.append({'case':b['meta'][i][0],'z':b['meta'][i][1],'boxes':bb.tolist(),'scores':ss.tolist(),'labels':cc.tolist(),'gt':b['boxes'][b['boxes'][:,0]==i,1:].tolist()})
    tp=np.diag(cm)[1:];dice=2*tp/np.maximum(cm.sum(0)[1:]+cm.sum(1)[1:],1)
    fg=cm[1:,1:].sum();fp=cm[0,1:].sum();fn=cm[1:,0].sum()
    return {'dice':dice.tolist(),'macro_dice':float(dice.mean()),'false_positive_pixels':int(fp),'foreground_recall':float(fg/max(fg+fn,1)),'foreground_precision':float(fg/max(fg+fp,1)),'detection':detection_metrics(records)}

def main():
    torch.set_num_threads(4);dest=ROOT/'salidas/clahe_revision';dest.mkdir(exist_ok=True)
    selected=json.loads((ROOT/'salidas/cierre/seleccion_modelo.json').read_text(encoding='utf-8'))
    model=PelvisSesion2();model.load_state_dict(torch.load(ROOT/selected['checkpoint'],weights_only=False,map_location='cpu')['model'])
    ck=torch.load(ROOT/'salidas/cajas_revision/rpn_grid/best_diagnostic.pth',weights_only=False,map_location='cpu')
    head=RPNGridHead(ck['config']['anchor_sizes']);head.load_state_dict(ck['head']);head.eval()
    val=Samples(ROOT/'salidas/revision_oct08/datos','val');loader=DataLoader(val,batch_size=8,collate_fn=collate)
    rows=[]
    for flag in (False,True):
        start=time.perf_counter();m=evaluate(model,head,loader,flag)
        row={'clahe':flag,'metrics':m,'seconds':time.perf_counter()-start};rows.append(row);print(json.dumps(row),flush=True)
    report={'scope':'120 validation slices, same trained weights, recalculated RPN and grid; inference-only preprocessing comparison; no retraining or test','clip_limit':2.,'tile_grid':[8,8],'window_level':400,'window_width':1800,'variants':rows}
    (dest/'inferencia.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
