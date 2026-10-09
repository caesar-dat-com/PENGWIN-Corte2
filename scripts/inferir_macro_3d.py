"""Pipeline 3D sobre TODOS los cortes contiguos. GT solo para evaluar tras inferencia."""
import sys,json,argparse,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import cv2
import SimpleITK as sitk
from pengwin.io import resolver_volumen,ventana_hu,verificar_alineacion,normalizar_lps
from pengwin.refinamiento_macro import refinar_probabilidades,control_calidad
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.fragmentos_sesion2 import reconstruir_fragmentos
from entrenar_pacientes import decode
from pengwin.metrics_avance3 import gate_semantic
from pengwin.instances import reconstruct_instances,separation_distances,match_instances

def json_save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def reference_grid(image):
    """256², mismo número de cortes y campo físico; centro del píxel coincide con resize lineal."""
    nx,ny,nz=image.GetSize();sx,sy,sz=image.GetSpacing()
    direction=np.asarray(image.GetDirection()).reshape(3,3)
    if not np.allclose(direction.T@direction,np.eye(3),atol=1e-5):
        raise ValueError('EDT requiere ejes ortogonales; re-muestrear geometría con cizallamiento antes de medir')
    ref=sitk.Image([256,256,nz],sitk.sitkUInt16)
    ref.SetSpacing((sx*nx/256,sy*ny/256,sz));ref.SetDirection(image.GetDirection())
    offset=np.array([(nx/256-1)*sx/2,(ny/256-1)*sy/2,0.])
    ref.SetOrigin(tuple(np.asarray(image.GetOrigin())+np.asarray(image.GetDirection()).reshape(3,3)@offset))
    return ref

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--output',type=Path,default=ROOT/'salidas/macro_sobel/volumenes');p.add_argument('--compare-gt',action='store_true');p.add_argument('--include-test',action='store_true');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--min-seed-mm3',type=float,default=50.)
    a=p.parse_args();torch.set_num_threads(a.threads)
    selected=json.loads((ROOT/'salidas/macro_sobel/seleccion.json').read_text())
    if a.checkpoint is None:a.checkpoint=ROOT/selected['checkpoint']
    splits=json.loads((ROOT/'splits/splits.json').read_text());cid=a.case
    split=next((k for k in ('train','val','test') if cid in splits[k]),None)
    if split is None:raise ValueError('Caso fuera del split congelado')
    if split=='test' and not a.include_test:raise ValueError('Test bloqueado hasta congelar el protocolo: requiere --include-test')
    dest=a.output/cid
    if dest.exists():raise ValueError('Salida ya existente; conserve evidencia o cambie carpeta')
    cfg=json.loads((ROOT/'config_datos.local.json').read_text())
    image=normalizar_lps(sitk.ReadImage(str(resolver_volumen(Path(cfg['images']),cid))))
    x=sitk.GetArrayFromImage(image);ref=reference_grid(image);sp=ref.GetSpacing()[::-1]
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=selected['checkpoint_sha256']:
        raise ValueError('Checkpoint distinto del seleccionado; volver a evaluar antes de reutilizar el refinamiento')
    policy={"conf":.25,"nms":.4,"boundary":.5,"interior":.5,"min_seed_mm3":a.min_seed_mm3,"refinement_strength":selected["strength"],"refinement_dimension":"2D, vecindad8, bilateral3x3; instancias3D vecindad26"}
    model=PelvisSesion2();model.load_state_dict(ck['model']);model.eval()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    sem=np.zeros((len(x),256,256),np.uint8);edge=np.zeros_like(sem,dtype=np.float32);cores=np.zeros_like(edge);records=[];times=[]
    with torch.inference_mode():
        warm=torch.zeros(1,3,256,256,device=device)
        for _ in range(3):model(warm)
        for z in range(len(x)):
            inp=torch.from_numpy(cv2.resize(ventana_hu(x[z]),(256,256),interpolation=cv2.INTER_AREA))[None,None].repeat(1,3,1,1).to(device)
            if device.type=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();out=model(inp);b,s,c=decode(out,conf=.25)
            b,s,c=(v.cpu().numpy() for v in (b,s,c))
            prob=refinar_probabilidades(out['mascaras'][0].softmax(0).cpu().numpy(),inp[0,0].cpu().numpy(),strength=selected['strength'])
            sem[z]=gate_semantic(prob,b,c)
            edge[z]=out['bordes'][0].sigmoid().cpu().numpy()
            cores[z]=out['interiores'][0].sigmoid().cpu().numpy()
            if device.type=='cuda':torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
            records.append({'z':z,'boxes':b.tolist(),'scores':s.tolist(),'labels':c.tolist()})
            if z%100==0:print(f'{cid}: corte {z}/{len(x)}',flush=True)
    del x
    pred,mapping=reconstruir_fragmentos(sem,edge,cores,sp,seed_min_volume_mm3=a.min_seed_mm3)
    distances=separation_distances(pred,mapping,sp)
    dest.mkdir(parents=True)
    for name,arr in [('instancias',pred),('semantica',sem)]:
        img=sitk.GetImageFromArray(arr);img.CopyInformation(ref);sitk.WriteImage(img,str(dest/f'{name}.mha'),True)
    json_save(dest/'control_calidad.json',control_calidad(sem))
    json_save(dest/'boxes.json',records);json_save(dest/'instancias.json',mapping);json_save(dest/'distancias.json',distances)
    metadata={'case':cid,'split':split,'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'epoch':ck['epoch'],
              'slices_contiguous':len(sem),'size_xyz':list(ref.GetSize()),'spacing_xyz':list(ref.GetSpacing()),'origin':list(ref.GetOrigin()),'direction':list(ref.GetDirection()),
              'device':str(device),'postprocessing':policy,'latency_mean_ms':float(np.mean(times)),'latency_median_ms':float(np.median(times)),
              'latency_scope':'forward+decode+NMS+refinamiento bilateral+semantic gating+transfer to CPU, 3 warmups; excluye lectura, resize y watershed 3D',
              'distance_definition':'EDT entre centros de voxeles de superficie, mm; aproximación discreta, no distancia exacta entre caras',
              'instance_method':'watershed 3D con interiores y bordes aprendidos dentro de región y cajas propias; semillas26, interior>0.5 y borde<0.5; mínimo20mm3; filtrado de semillas según min_seed_mm3',
              'geometry_note':'Resolución XY reducida a 256; spacing/origen ajustados al campo físico. No se apilan cortes salteados.'}
    if a.compare_gt:
        gtimage=normalizar_lps(sitk.ReadImage(str(resolver_volumen(Path(cfg['labels']),cid))))
        verificar_alineacion(image,gtimage,cid)
        gt=sitk.GetArrayFromImage(sitk.Resample(gtimage,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8))
        from pengwin.instances import semantic_labels
        gs=semantic_labels(gt);cm=np.bincount((gs.astype(np.int64)*4+sem).ravel(),minlength=16).reshape(4,4)
        dice=2*np.diag(cm)[1:]/np.maximum(cm.sum(0)[1:]+cm.sum(1)[1:],1)
        metadata['semantic']={'dice':dice.tolist(),'macro_dice':float(dice.mean())}
        gm={int(i):(int(i)-1)//10+1 for i in np.unique(gt) if i>0}
        gd=separation_distances(gt,gm,sp,principal_ids={1:1,2:11,3:21});matches=match_instances(pred,gt,mapping)
        pairs={m['pred']:m['gt'] for m in matches if m['pred'] is not None and m['gt'] is not None and m['iou']>=.5}
        metadata['distance_matching_min_iou']=.5
        gtlookup={d['instance']:d for d in gd};comparison=[]
        for d in distances:
            g=gtlookup.get(pairs.get(d['instance']));sameprincipal=g is not None and pairs.get(d['principal'])==g['principal']
            comparison.append({'pred_instance':d['instance'],'matched_gt':pairs.get(d['instance']),
                               'principal_matches':sameprincipal,'pred_mm':d['distance_surface_voxel_centers_mm'],
                               'gt_mm':g['distance_surface_voxel_centers_mm'] if sameprincipal else None,
                               'absolute_error_mm':abs(d['distance_surface_voxel_centers_mm']-g['distance_surface_voxel_centers_mm']) if sameprincipal else None})
        errors=[d['absolute_error_mm'] for d in comparison if d['absolute_error_mm'] is not None]
        gtrows=[m for m in matches if m['gt'] is not None]
        metadata['fragment_metrics']={'gt_fragments':len(gm),'pred_instances':len(mapping),'dice_gt_macro_with_misses':float(np.mean([m['dice'] for m in gtrows])) if gtrows else None,'iou_gt_macro_with_misses':float(np.mean([m['iou'] for m in gtrows])) if gtrows else None,'dice_symmetric_macro_with_unmatched':float(np.mean([m['dice'] for m in matches])) if matches else None,'missed_gt':sum(m['pred'] is None for m in matches),'extra_pred':sum(m['gt'] is None for m in matches),'distance_comparable_pairs':len(errors),'distance_MAE_mm':float(np.mean(errors)) if errors else None}
        json_save(dest/'emparejamientos.json',matches);json_save(dest/'distancias_gt.json',gd);json_save(dest/'comparacion_distancias.json',comparison)
    json_save(dest/'resumen.json',metadata)
    print(json.dumps(metadata,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
