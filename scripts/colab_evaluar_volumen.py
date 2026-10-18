"""Evaluación volumétrica para Colab: sin candados de sha/protocolo, todo en memoria.

Por cada caso: inferencia corte a corte (en lotes) -> semántica con compuerta de
cajas -> control de anomalías (con y sin corrección) -> instancias 3D ->
emparejamiento contra GT -> distancias fragmento-principal. Compara la salida
cruda del modelo contra la corregida para medir qué aporta el control.
"""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch,SimpleITK as sitk,cv2
from pengwin.io import resolver_volumen,normalizar_lps,ventana_hu,rutas_dataset
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.metrics_avance3 import gate_semantic
from pengwin.instancias_estables import reconstruir,resumen_fragmentos
from pengwin.instances import match_instances,separation_distances,semantic_labels
from pengwin import anomalias
from entrenar_sesion2 import decode
from inferir_macro_3d import reference_grid

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=float),encoding='utf-8')

@torch.inference_mode()
def predecir(model,x_hu,device,margen,lote=16,conf=.25):
    import time
    n=len(x_hu);sem=np.zeros((n,256,256),np.uint8);edge=np.zeros((n,256,256),np.float32);core=np.zeros_like(edge)
    gray=np.zeros_like(sem);boxes=[];times=[]
    hu=np.stack([cv2.resize(s.astype(np.float32),(256,256),interpolation=cv2.INTER_AREA) for s in x_hu])
    for i in range(0,n,lote):
        batch_slices=[cv2.resize(ventana_hu(s),(256,256),interpolation=cv2.INTER_AREA) for s in x_hu[i:i+lote]]
        for k,im_s in enumerate(batch_slices):
            gray[i+k]=np.rint(np.clip(im_s,0,1)*255).astype(np.uint8)
        im=torch.from_numpy(np.stack(batch_slices))[:,None].repeat(1,3,1,1).to(device)
        if device.type=='cuda':torch.cuda.synchronize()
        t0=time.perf_counter();out=model(im);prob=out['mascaras'].float().softmax(1).cpu().numpy()
        for k in range(len(im)):
            bb,ss,cc=[v.cpu().numpy() for v in decode(out,k,conf)]
            sem[i+k]=gate_semantic(prob[k],bb,cc,margen)
            boxes.append({'z':int(i+k),'boxes':bb.tolist(),'scores':ss.tolist(),'labels':cc.tolist()})
        edge[i:i+lote]=out['bordes'].float().sigmoid().cpu().numpy();core[i:i+lote]=out['interiores'].float().sigmoid().cpu().numpy()
        if device.type=='cuda':torch.cuda.synchronize()
        dt_ms=(time.perf_counter()-t0)*1000.0/len(im)
        times.extend([dt_ms]*len(im))
    return sem,edge,core,hu,gray,boxes,times

def evaluar(sem,edge,core,gt,sp,policy,min_iou):
    pred,mapping=reconstruir(sem,edge,core,sp,policy);matches=match_instances(pred,gt,mapping);r=resumen_fragmentos(matches)
    gtsem=semantic_labels(gt);cm=np.bincount((gtsem.astype(np.int64)*4+sem).ravel(),minlength=16).reshape(4,4)
    r['semantic_dice']=(2*np.diag(cm)[1:]/np.maximum(cm.sum(0)[1:]+cm.sum(1)[1:],1)).tolist();r['confusion']=cm.tolist()
    gm={int(i):(int(i)-1)//10+1 for i in np.unique(gt) if i};principals={1:1,2:11,3:21}
    gm={i:c for i,c in gm.items() if principals[c] in gm}
    dist=separation_distances(pred,mapping,sp)
    gd_list=separation_distances(gt,gm,sp,principal_ids={c:principals[c] for c in set(gm.values())}) if gm else []
    gd={d['instance']:d for d in gd_list}
    pair={m['pred']:m['gt'] for m in matches if m['pred'] is not None and m['gt'] is not None and m['iou']>=min_iou}
    comp=[]
    for d in dist:
        g_id=pair.get(d['instance']);g=gd.get(g_id)
        valid=bool(g is not None and pair.get(d['principal'])==g['principal'])
        err_val=abs(d['distance_surface_voxel_centers_mm']-g['distance_surface_voxel_centers_mm']) if valid else None
        comp.append({'pred_instance':int(d['instance']),'matched_gt':int(g_id) if g_id is not None else None,
                     'valid_pair':valid,'pred_mm':float(d['distance_surface_voxel_centers_mm']),
                     'gt_mm':float(g['distance_surface_voxel_centers_mm']) if valid else None,
                     'absolute_error_mm':float(err_val) if err_val is not None else None})
    err=[c['absolute_error_mm'] for c in comp if c['valid_pair']]
    r.update(distance_min_iou=min_iou,distance_valid_pairs=len(err),distance_MAE_mm=float(np.mean(err)) if err else None)
    return r,pred,mapping,matches,dist,gd_list,comp

def main():
    import hashlib
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gate-margin',type=float,default=.1);p.add_argument('--min-iou',type=float,default=.5)
    p.add_argument('--referencia',type=Path,help='JSON de anomalias.referencia_desde_gt (train)')
    p.add_argument('--policy',default='{"method":"interfaces","threshold":0.5,"seed_mm3":50.0,"min_volume_mm3":500.0,"fusionar":false}');a=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    ck_sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    splits=json.loads((ROOT/'splits/splits.json').read_text(encoding='utf-8'))
    model=PelvisSesion2(canales=ck.get('config',{}).get('canales',8));model.load_state_dict(ck['model']);model.to(device).eval()
    images,labels=rutas_dataset();policy=json.loads(a.policy);ref=json.loads(a.referencia.read_text()) if a.referencia else None
    if ref:ref={int(k):v for k,v in ref.items()}
    a.output.mkdir(parents=True,exist_ok=True);tabla=[]
    for cid in a.cases:
        split_name=next((k for k in ('train','val','test') if cid in splits[k]),'val')
        image=normalizar_lps(sitk.ReadImage(str(resolver_volumen(images,cid))));x=sitk.GetArrayFromImage(image);grid=reference_grid(image)
        entrada=anomalias.anomalias_entrada(image,x)
        sem,edge,core,hu,gray,boxes,times=predecir(model,x,device,a.gate_margin)
        gtimg=normalizar_lps(sitk.ReadImage(str(resolver_volumen(labels,cid))))
        gt=sitk.GetArrayFromImage(sitk.Resample(gtimg,grid,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8));sp=grid.GetSpacing()[::-1]
        crudo,_,_,_,_,_,_=evaluar(sem,edge,core,gt,sp,policy,a.min_iou)
        fixed,informe=anomalias.controlar(sem,sp,hu=hu,referencia=ref)
        corr,pred,mapping,matches,dist,gd_list,comp=evaluar(fixed,edge,core,gt,sp,policy,a.min_iou)
        _,inf2=anomalias.controlar(fixed,sp,hu=hu,referencia=ref,instancias=pred,mapping=mapping,corregir=False)
        informe['anomalias_post']=inf2['anomalias'];informe['entrada']=entrada
        d=a.output/cid;d.mkdir(exist_ok=True)
        for name,arr in [('semantica_cruda',sem),('semantica_corregida',fixed),('semantica',fixed),
                         ('instancias',pred.astype(np.uint16)),('gt',gt),('gt_evaluacion',gt),('ct_ventana',gray)]:
            o=sitk.GetImageFromArray(arr);o.CopyInformation(grid);sitk.WriteImage(o,str(d/f'{name}.mha'),True)
        save(d/'boxes.json',boxes)
        save(d/'instancias.json',{str(k):int(v) for k,v in mapping.items()})
        save(d/'distancias.json',dist)
        save(d/'distancias_gt.json',gd_list)
        save(d/'emparejamientos.json',matches)
        save(d/'comparacion_distancias.json',comp)
        save(d/'prediccion.json',{
            'case':cid,'split':split_name,'sha256':ck_sha,'epoch':int(ck.get('epoch',0)),
            'detector':'PelvisSesion2 (GPU + control de anomalías)','slices':len(x),'device':str(device),
            'latency_mean_ms':float(np.mean(times)),'latency_p50_ms':float(np.median(times)),'latency_p95_ms':float(np.percentile(times,95)),
            'spacing_xyz':list(grid.GetSpacing()),'origin':list(grid.GetOrigin()),'direction':list(grid.GetDirection())
        })
        save(d/'anomalias.json',informe)
        resumen_obj={**corr,'case':cid,'policy':policy,'crudo':crudo,'corregido':corr,'checkpoint_sha256':ck_sha,'epoch':int(ck.get('epoch',0))}
        save(d/'resumen.json',resumen_obj)
        fila={'case':cid,'anomalias':len(informe['anomalias']),'estado':informe['estado']}
        for k,r in (('crudo',crudo),('corr',corr)):
            fila.update({f'{k}_dice_SA':r['semantic_dice'][0],f'{k}_dice_LI':r['semantic_dice'][1],f'{k}_dice_RI':r['semantic_dice'][2],
                         f'{k}_pred':r['pred'],f'{k}_extra':r['extra'],f'{k}_missed':r['missed'],f'{k}_dice_sim':r['dice_symmetric'],f'{k}_pares':r['distance_valid_pairs'],f'{k}_MAE':r['distance_MAE_mm']})
        fila['gt']=crudo['gt'];tabla.append(fila);print('CASO',json.dumps(fila,default=float),flush=True)
    save(a.output/'tabla.json',tabla)

if __name__=='__main__':main()
