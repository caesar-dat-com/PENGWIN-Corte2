"""Selección de posprocesado sobre 3 volúmenes val; protocolo congelado para test."""
import sys,json,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,SimpleITK as sitk
from pengwin.io import resolver_volumen,normalizar_lps
from pengwin.instancias_estables import reconstruir,resumen_fragmentos
from pengwin.instances import match_instances,separation_distances,semantic_labels
from pengwin.refinamiento_macro import control_calidad

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def load_case(cid, results_root='salidas/cierre/volumenes'):
    dest=ROOT/results_root/cid;ref=sitk.ReadImage(str(dest/'semantica.mha'))
    sem=sitk.GetArrayFromImage(ref);edge=sitk.GetArrayFromImage(sitk.ReadImage(str(dest/'bordes.mha')));core=sitk.GetArrayFromImage(sitk.ReadImage(str(dest/'interiores.mha')))
    cfg=json.loads((ROOT/'config_datos.local.json').read_text(encoding='utf-8'));gtimage=normalizar_lps(sitk.ReadImage(str(resolver_volumen(Path(cfg['labels']),cid))))
    gt=sitk.GetArrayFromImage(sitk.Resample(gtimage,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8))
    return dest,ref,sem,edge,core,gt

def evaluate(cid,policy,write=False,results_root='salidas/cierre/volumenes'):
    dest,ref,sem,edge,core,gt=load_case(cid,results_root);sp=ref.GetSpacing()[::-1]
    pred,mapping=reconstruir(sem,edge,core,sp,policy);matches=match_instances(pred,gt,mapping);summary=resumen_fragmentos(matches)
    if not write:return summary
    distances=separation_distances(pred,mapping,sp)
    gm={int(i):(int(i)-1)//10+1 for i in np.unique(gt) if i};principals={1:1,2:11,3:21}
    missing=[r for r in (1,2,3) if r in gm.values() and principals[r] not in gm]
    gd=separation_distances(gt,{i:r for i,r in gm.items() if r not in missing},sp,principal_ids=principals)
    pair={r['pred']:r['gt'] for r in matches if r['pred'] is not None and r['gt'] is not None and r['iou']>=.5}
    lookup={r['instance']:r for r in gd};comparison=[]
    for d in distances:
        g=lookup.get(pair.get(d['instance']));valid=g is not None and pair.get(d['principal'])==g['principal']
        comparison.append({'instance':d['instance'],'gt_instance':pair.get(d['instance']),'valid_pair':valid,'pred_mm':d['distance_surface_voxel_centers_mm'],'gt_mm':g['distance_surface_voxel_centers_mm'] if valid else None,'absolute_error_mm':abs(d['distance_surface_voxel_centers_mm']-g['distance_surface_voxel_centers_mm']) if valid else None})
    errors=[r['absolute_error_mm'] for r in comparison if r['valid_pair']]
    summary.update(distance_valid_pairs=len(errors),distance_MAE_mm=float(np.mean(errors)) if errors else None,missing_gt_principal_regions=missing)
    gtsem=semantic_labels(gt);cm=np.bincount((gtsem.astype(np.int64)*4+sem).ravel(),minlength=16).reshape(4,4);dice=2*np.diag(cm)[1:]/np.maximum(cm.sum(0)[1:]+cm.sum(1)[1:],1)
    summary['semantic_dice']=dice.tolist();summary['case']=cid;summary['policy']=policy
    for name,arr in [('instancias',pred),('gt_evaluacion',gt)]:
        obj=sitk.GetImageFromArray(arr);obj.CopyInformation(ref);sitk.WriteImage(obj,str(dest/f'{name}.mha'),True)
    save(dest/'instancias.json',mapping);save(dest/'distancias.json',distances);save(dest/'distancias_gt.json',gd);save(dest/'comparacion_distancias.json',comparison);save(dest/'emparejamientos.json',matches);save(dest/'resumen.json',summary);save(dest/'control_calidad.json',control_calidad(sem))
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');a=p.parse_args();out=ROOT/'salidas/cierre'
    splits=json.loads((ROOT/'splits/splits.json').read_text(encoding='utf-8'))
    if a.test:
        frozen=json.loads((out/'protocolo_final.json').read_text(encoding='utf-8'));cases=frozen['test_volume_cases'];policy=frozen['instance_policy']
        results=[evaluate(cid,policy,True) for cid in cases];save(out/'evaluacion_volumetrica_test.json',{'scope':'3 pacientes test predefinidos; no ajustar tras ver estos resultados','cases':results});print('TEST 3D',results,flush=True);return
    if (out/'protocolo_final.json').exists():raise ValueError('Protocolo ya congelado; no recalibrar esta ejecución')
    cases=splits['val'][:3]
    policies=[{'method':'interiores','seed_mm3':50.,'min_volume_mm3':20.},{'method':'componentes','min_volume_mm3':20.}]+[{'method':'interfaces','threshold':t,'seed_mm3':50.,'min_volume_mm3':20.} for t in (.3,.5,.7)]+[{'method':'interfaces','threshold':.5,'seed_mm3':50.,'min_volume_mm3':v} for v in (100.,500.)]
    scores=[]
    for policy in policies:
        results=[evaluate(cid,policy) for cid in cases]
        score=float(np.mean([r['dice_symmetric'] or 0 for r in results]));scores.append({'policy':policy,'score':score,'cases':results});print('POLITICA',policy,score,flush=True)
    selected=max(scores,key=lambda r:r['score']);save(out/'calibracion_instancias.json',{'scope':'Tres primeros pacientes val, mismos casos en siete variantes; no número de fragmentos forzado; no GT modificado; filtrar predicciones por volumen puede eliminar fragmentos verdaderos pequeños','candidates':scores,'selected':selected})
    model=json.loads((out/'seleccion_modelo.json').read_text(encoding='utf-8'))
    frozen={'checkpoint':model['checkpoint'],'checkpoint_sha256':model['sha256'],'instance_policy':selected['policy'],'refinement_strength':0.,'conf':.25,'nms':.4,'distance_min_iou':.5,'test_volume_cases':splits['test'][:3],'test_slice_cases':splits['test'],'slice_fractions':[.1,.5,.9],'sam_fractions':[.5],'test_history':'El conjunto test tuvo exposición exploratoria en avances anteriores. Este protocolo no se ajusta con los resultados nuevos; no es un test virgen.','dataset_split_sha256':hashlib.sha256((ROOT/'splits/splits.json').read_bytes()).hexdigest()}
    save(out/'protocolo_final.json',frozen)
    results=[evaluate(cid,selected['policy'],True) for cid in cases];save(out/'evaluacion_volumetrica_val.json',{'scope':'Validación usada para seleccionar; no test','cases':results})
    print('SELECCION INSTANCIAS',selected['policy'],flush=True)
if __name__=='__main__':main()
