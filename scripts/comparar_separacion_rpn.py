"""Comparación acotada de separación 3D, exclusivamente en validación."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from evaluar_instancias_cierre import load_case
from pengwin.instancias_estables import reconstruir,resumen_fragmentos
from pengwin.instances import match_instances

def main():
    dest=ROOT/'salidas/cajas_revision/separacion';dest.mkdir(exist_ok=True)
    policies=[
        {'method':'interfaces','threshold':.3,'seed_mm3':50.,'min_volume_mm3':500.},
        {'method':'interfaces','threshold':.7,'seed_mm3':50.,'min_volume_mm3':500.},
        {'method':'interfaces','threshold':.5,'seed_mm3':200.,'min_volume_mm3':500.},
        {'method':'interiores','seed_mm3':50.,'min_volume_mm3':500.},
        {'method':'componentes','min_volume_mm3':500.}]
    baseline=[];results=[{'policy':p,'cases':[]} for p in policies]
    for cid in ('002','012','028'):
        folder,ref,sem,edge,core,gt=load_case(cid,'salidas/cajas_revision/volumenes')
        baseline.append(json.loads((folder/'resumen.json').read_text(encoding='utf-8')))
        for variant in results:
            start=time.perf_counter()
            pred,mapping=reconstruir(sem,edge,core,ref.GetSpacing()[::-1],variant['policy'])
            summary=resumen_fragmentos(match_instances(pred,gt,mapping));summary['case']=cid
            variant['cases'].append(summary)
            print(cid,variant['policy'],summary,'seconds',round(time.perf_counter()-start,1),flush=True)
            (dest/'progress.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    def aggregate(rows):
        return {**{key:float(np.mean([r[key] for r in rows])) for key in ('dice_gt','iou_gt','dice_symmetric')},**{key:sum(r[key] for r in rows) for key in ('missed','extra','matches_iou05')}}
    base=aggregate(baseline)
    for variant in results:
        m=aggregate(variant['cases']);variant['aggregate']=m
        variant['eligible']=m['dice_gt']>base['dice_gt'] and m['dice_symmetric']>base['dice_symmetric'] and m['missed']<=base['missed'] and m['extra']<=base['extra'] and m['matches_iou05']>=base['matches_iou05'] and all(n['dice_gt']>=b['dice_gt']-.005 for n,b in zip(variant['cases'],baseline))
    eligible=[v for v in results if v['eligible']]
    selected=max(eligible,key=lambda v:v['aggregate']['dice_symmetric']) if eligible else None
    report={'scope':'Three validation volumes, fixed predictions and minimum volume; no GT used by reconstruction; no test tuning','baseline':base,'baseline_cases':baseline,'candidates':results,'selected':selected}
    (dest/'comparacion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SELECTED',json.dumps(selected),flush=True)

if __name__=='__main__':main()
