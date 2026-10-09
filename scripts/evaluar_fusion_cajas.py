"""Evaluación de respaldo grid y propuestas RPN, exclusivamente sobre val."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from pengwin.propuestas_regionales import combinar_regiones
from pengwin.metrics_avance3 import detection_metrics,gate_semantic,box_iou
from entrenar_sesion2 import Samples

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def main():
    torch.set_num_threads(4);out=ROOT/'salidas/cajas_revision';old=read(out/'detecciones_val.json');new=read(out/'rpn_grid/detecciones_diagnostic.json');base=read(out/'auditoria.json')
    val=torch.load(ROOT.parent/'.cajas-cache/val.pth',weights_only=False);ds=Samples(ROOT/'salidas/revision_oct08/datos','val');truth=np.stack([ds[i]['semantic'].numpy() for i in range(len(ds))]);del ds
    results=[];chosen=None
    for padding in (0.,.005,.01,.015):
        records=[];coverage=[];cm=np.zeros((4,4),np.int64);missed=0;fallback=0
        for i,(o,n) in enumerate(zip(old,new)):
            assert (o['case'],o['z'])==(n['case'],n['z'])==tuple(val['meta'][i])
            bb,ss,cc,sources=combinar_regiones(o['boxes'],o['scores'],o['labels'],n['boxes'],n['scores'],n['labels'],padding=padding)
            fallback+=sources.count('grid_respaldo');records.append({'case':o['case'],'z':o['z'],'gt':o['gt'],'boxes':bb.tolist(),'scores':ss.tolist(),'labels':cc.tolist(),'sources':sources})
            gt=truth[i];pr=gate_semantic(val['probabilities'][i].float().numpy(),bb,cc);cm+=np.bincount((gt*4+pr).ravel(),minlength=16).reshape(4,4)
            for g in o['gt']:
                mask=gt==int(g[0])+1;inside=np.zeros(mask.shape,bool);boxes=bb[cc==int(g[0])];missed+=int(len(boxes)==0)
                for b in boxes:
                    x1,y1=np.floor(b[:2]*256).astype(int);x2,y2=np.ceil(b[2:]*256).astype(int);inside[y1:y2,x1:x2]=True
                coverage.append(float((inside&mask).sum()/max(mask.sum(),1)))
        dice=2*np.diag(cm)[1:]/np.maximum(cm.sum(0)[1:]+cm.sum(1)[1:],1);det=detection_metrics(records);cov=float(np.mean(coverage))
        metrics={'detection':det,'semantic_dice':dice.tolist(),'macro_dice':float(dice.mean()),'gt_mask_coverage':cov,'missed_regions':missed,'duplicate_regions':0,'fallback_boxes':fallback}
        score=.5*det['mAP50_95']+.25*det['mAP50']+.25*cov
        eligible=det['mAP50']>=base['detection']['mAP50'] and det['mean_gt_iou_at_conf025']>=base['detection']['mean_gt_iou_at_conf025'] and cov>=base['mean_gt_mask_coverage'] and metrics['macro_dice']>=.6671909875850801 and missed<=base['regions_without_box']
        row={'padding':padding,'new_conf':.1,'old_conf':.25,'score':score,'eligible':eligible,'metrics':metrics};results.append(row)
        if eligible and (chosen is None or score>chosen['score']):chosen=row;save(out/'detecciones_fusion.json',records)
        print('RESPALDO',padding,'mAP',round(det['mAP50'],3),'IoU',round(det['mean_gt_iou_at_conf025'],3),'cobertura',round(cov,3),'Dice',round(metrics['macro_dice'],3),'omitidos',missed,eligible,flush=True)
    save(out/'fusion.json',{'scope':'Val, RPN con respaldo grid; nunca GT en inferencia; una bbox macro por región, no número impuesto de fragmentos','candidates':results,'selected':chosen})
if __name__=='__main__':main()
