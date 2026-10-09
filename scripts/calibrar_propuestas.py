"""Calibración de cobertura en validación, sin nuevos datos test."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from pengwin.models.rpn_grid import RPNGridHead
from entrenar_propuestas_grid import evaluate,save
from entrenar_sesion2 import Samples

def main():
    torch.set_num_threads(4);dest=ROOT/'salidas/cajas_revision/rpn_grid';ck=torch.load(dest/'best_diagnostic.pth',weights_only=False)
    head=RPNGridHead(ck['config']['anchor_sizes']);head.load_state_dict(ck['head']);head.eval()
    val=torch.load(ROOT.parent/'.cajas-cache/val.pth',weights_only=False);dataset=Samples(ROOT/'salidas/revision_oct08/datos','val');truth=np.stack([dataset[i]['semantic'].numpy() for i in range(len(dataset))])
    baseline=json.loads((ROOT/'salidas/cajas_revision/auditoria.json').read_text(encoding='utf-8'));rows=[];chosen=None
    for confidence in (.10,.15,.25):
        for padding in (0.,.01,.02):
            metrics,records=evaluate(head,val,truth,True,confidence,padding);det=metrics['detection']
            eligible=det['mAP50']>=baseline['detection']['mAP50'] and det['mean_gt_iou_at_conf025']>=baseline['detection']['mean_gt_iou_at_conf025'] and metrics['gt_mask_coverage']>=baseline['mean_gt_mask_coverage'] and metrics['macro_dice']>=.6671909875850801 and metrics['missed_regions']<=baseline['regions_without_box']
            score=.5*det['mAP50_95']+.25*det['mAP50']+.25*metrics['gt_mask_coverage'];row={'metrics':metrics,'eligible':eligible,'score':score};rows.append(row)
            if eligible and (chosen is None or score>chosen['score']):chosen=row;save(dest/'detecciones_seleccionadas.json',records)
            print('CALIBRACION',confidence,padding,'mAP',round(det['mAP50'],3),'IoU',round(det['mean_gt_iou_at_conf025'],3),'Dice',round(metrics['macro_dice'],3),'cobertura',round(metrics['gt_mask_coverage'],3),'omitidos',metrics['missed_regions'],'elegible',eligible,flush=True)
    save(dest/'calibracion.json',{'scope':'9 variantes prespecificadas, solo val; una caja macro por región anatómica, no un límite de fragmentos; padding fracción de imagen','candidates':rows,'selected':chosen})
if __name__=='__main__':main()
