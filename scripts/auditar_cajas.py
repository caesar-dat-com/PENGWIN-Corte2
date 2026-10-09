"""Diagnóstico del grid propio contra GT en train/val; no abre test."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch.utils.data import DataLoader
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.models.detector import decodificar_grid
from pengwin.metrics_avance3 import box_iou,detection_metrics
from entrenar_sesion2 import Samples,collate,decode

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    torch.set_num_threads(4);out=ROOT/'salidas/cajas_revision';out.mkdir(exist_ok=True)
    selected=json.loads((ROOT/'salidas/cierre/seleccion_modelo.json').read_text(encoding='utf-8'));checkpoint=ROOT/selected['checkpoint']
    model=PelvisSesion2();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.eval()
    cache=ROOT.parent/'.cajas-cache';cache.mkdir(exist_ok=True)
    rows=[];details=[];conversion_errors=[]
    with torch.inference_mode():
        for split in ('val','train'):
            dataset=Samples(ROOT/'salidas/revision_oct08/datos',split);loader=DataLoader(dataset,batch_size=8,collate_fn=collate)
            features=[];logits=[];boxes_list=[];metadata=[];probs=[]
            for batch in loader:
                pred=model(batch['x']);features.append(pred['features'].clone());logits.append(pred['clases'].clone())
                if split=='val':probs.append(pred['mascaras'].softmax(1).to(torch.float16))
                for i in range(len(batch['x'])):
                    gt=batch['boxes'][batch['boxes'][:,0]==i,1:].clone();boxes_list.append(gt);metadata.append(batch['meta'][i])
                    if split!='val':continue
                    bb,ss,cc=[x.numpy() for x in decode(pred,i,.05)];keep=ss>=.25
                    rows.append({'case':batch['meta'][i][0],'z':batch['meta'][i][1],'gt':gt.tolist(),'boxes':bb.tolist(),'scores':ss.tolist(),'labels':cc.tolist()})
                    for g in gt:
                        cls=int(g[0]);candidates=bb[keep&(cc==cls)];ious=box_iou(g[1:].numpy(),candidates)
                        area=(g[3]-g[1])*(g[4]-g[2]);coverage=[]
                        for box in candidates:
                            inter=np.maximum(0,np.minimum(box[2:],g[3:].numpy())-np.maximum(box[:2],g[1:3].numpy())).prod();coverage.append(float(inter/max(float(area),1e-9)))
                        mask=batch['semantic'][i].numpy()==cls+1;inside=np.zeros(mask.shape,bool)
                        for box in candidates:
                            x1,y1=np.floor(box[:2]*256).astype(int);x2,y2=np.ceil(box[2:]*256).astype(int);inside[y1:y2,x1:x2]=True
                        details.append({'case':batch['meta'][i][0],'z':batch['meta'][i][1],'region':cls+1,'candidates':len(candidates),'best_iou':float(ious.max()) if len(ious) else 0.,'bbox_coverage':max(coverage,default=0.),'gt_mask_coverage':float((inside&mask).sum()/max(mask.sum(),1))})
                        # Codificar GT ideal y decodificar: comprueba ejes, escala y centro.
                        grid=torch.full((8,16,16),-30.);cx=(g[1]+g[3])/2;cy=(g[2]+g[4])/2;ix=min(int(cx*16),15);iy=min(int(cy*16),15)
                        values=torch.stack([cx*16-ix,cy*16-iy,g[3]-g[1],g[4]-g[2]]).clamp(1e-6,1-1e-6)
                        grid[0,iy,ix]=30;grid[1:5,iy,ix]=torch.logit(values);grid[5+cls,iy,ix]=30
                        decoded,_,_=decodificar_grid(grid,.5);conversion_errors.append(float((decoded[0]-g[1:]).abs().max()))
            data={'features':torch.cat(features),'classes':torch.cat(logits),'boxes':boxes_list,'meta':metadata,'checkpoint_sha256':selected['sha256']}
            if split=='val':data['probabilities']=torch.cat(probs)
            torch.save(data,cache/f'{split}.pth');print('CACHE',split,len(metadata),flush=True)
    summary={'scope':'120 cortes val, no test; cache backbone congelado train/val para investigar cajas','architecture':'Grid propio + NMS propio; no hay módulo RPN en PelvisSesion2','checkpoint_sha256':selected['sha256'],'coordinate_roundtrip_max_normalized_error':max(conversion_errors),'regions_evaluated':len(details),'regions_with_duplicate_boxes':sum(r['candidates']>1 for r in details),'regions_without_box':sum(r['candidates']==0 for r in details),'regions_mask_coverage_below_95pct':sum(r['gt_mask_coverage']<.95 for r in details),'mean_gt_mask_coverage':float(np.mean([r['gt_mask_coverage'] for r in details])),'detection':detection_metrics(rows)}
    save(out/'auditoria.json',summary);save(out/'detecciones_val.json',rows);save(out/'cobertura_val.json',details);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
