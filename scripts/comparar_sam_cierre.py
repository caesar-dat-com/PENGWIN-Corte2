"""Evaluación congelada: detección en 3 cortes por test, SAM en 1 corte central por test.

Se reutiliza el muestreo 10/50/90% del avance 2. Incluye negativos y omisiones.
SAM recibe exclusivamente cajas del modelo, nunca máscaras o cajas GT.
"""
import argparse,json,sys,hashlib,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import cv2,numpy as np,torch,SimpleITK as sitk
from segment_anything import sam_model_registry,SamPredictor
from pengwin.io import resolver_volumen,ventana_hu,verificar_alineacion
from pengwin.dataset import extraer_bboxes_region
from pengwin.instances import semantic_labels,reconstruct_instances,match_instances
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.instancias_estables import reconstruir
from entrenar_pacientes import decode
from pengwin.metrics_avance3 import detection_metrics,classification_metrics,gate_semantic

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def semantic_metrics(cm):
    inter=np.diag(cm)[1:];gt=cm.sum(1)[1:];pr=cm.sum(0)[1:];den=gt+pr
    return {'dice_per_region':(2*inter/np.maximum(den,1)).tolist(),'iou_per_region':(inter/np.maximum(den-inter,1)).tolist(),
            'macro_dice':float(np.mean(2*inter/np.maximum(den,1))),'macro_iou':float(np.mean(inter/np.maximum(den-inter,1))),'confusion':cm.tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--sam',type=Path,default=Path(json.loads((ROOT/'config_datos.local.json').read_text(encoding='utf-8'))['sam_checkpoint']));p.add_argument('--output',type=Path,default=ROOT/'salidas/cierre/comparacion_test')
    p.add_argument('--include-test',action='store_true');p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    frozen_policy=json.loads((ROOT/'salidas/cierre/protocolo_final.json').read_text(encoding='utf-8'))
    if a.checkpoint is None:a.checkpoint=ROOT/frozen_policy['checkpoint']
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=frozen_policy['checkpoint_sha256']:raise ValueError('Modelo diferente del protocolo congelado')
    if not a.include_test:raise ValueError('Se requiere --include-test: modelo y protocolo deben estar congelados')
    if a.output.exists():raise ValueError('No sobrescribir ni repetir ajuste con test; carpeta ya existe')
    run_config=a.checkpoint.parent/'config.json'
    if run_config.exists() and json.loads(run_config.read_text(encoding='utf-8'))['status']!='completed':raise ValueError('No abrir test mientras se entrena')
    torch.set_num_threads(a.threads);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    policy={"conf":.25,"nms":.4,"boundary":.5}
    model=PelvisSesion2();model.load_state_dict(ck['model']);model.to(device).eval()
    sam=sam_model_registry['vit_b'](checkpoint=str(a.sam)).to(device).eval();predictor=SamPredictor(sam)
    cfg=json.loads((ROOT/'config_datos.local.json').read_text(encoding='utf-8'));splits=json.loads((ROOT/'splits/splits.json').read_text(encoding='utf-8'))
    a.output.mkdir(parents=True);shutil.copy2(a.checkpoint,a.output/'modelo_congelado.pth')
    frozen={'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'sam_sha256':hashlib.sha256(a.sam.read_bytes()).hexdigest(),
            'test_patients':splits['test'],'z_fractions':[.1,.5,.9],'sam_z_fractions':[.5],'conf':.25,'ap_floor':.05,'nms':.4,'instance_boundary':.5,
            'note':'Test tuvo exposición exploratoria en avance 2; no se usó para elegir esta ejecución. Esta evaluación no mide todo el volumen.',
            'instance_metrics':'Instancias 2D por corte, asignación 1:1 IoU dentro de región; no equivalen a fragmentos 3D',
            'SAM_prompts':'Todas las cajas predichas tras NMS, conf>=.25, multimask_output=False, zero-shot',
            'device':str(device),'postprocessing':policy,'instance_2d_policy':'Propio: metodo3D elegido, umbrales2D fijos semilla64px e instancia16px. SAM: componentes8 e instancia16px. No equivale a fragmentos volumetricos.'}
    save(a.output/'protocolo_congelado.json',frozen)
    matrices={k:np.zeros((4,4),np.int64) for k in ('propio','sam')};instance_rows={k:[] for k in matrices};records=[];truth=[];probs=[];timings=[]
    with torch.inference_mode():
        for cid in splits['test']:
            img=sitk.DICOMOrient(sitk.ReadImage(str(resolver_volumen(Path(cfg['images']),cid))),'LPS')
            label=sitk.DICOMOrient(sitk.ReadImage(str(resolver_volumen(Path(cfg['labels']),cid))),'LPS');verificar_alineacion(img,label,cid)
            x,y=sitk.GetArrayFromImage(img),sitk.GetArrayFromImage(label)
            for z in np.unique(np.rint(np.array([.1,.5,.9])*(len(x)-1)).astype(int)):
                gray=cv2.resize(ventana_hu(x[z]),(256,256),interpolation=cv2.INTER_AREA);ids=cv2.resize(y[z].astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST)
                inp=torch.from_numpy(gray)[None,None].repeat(1,3,1,1).to(device);out=model(inp)
                b,s,c=decode(out,conf=.05);b,s,c=(v.cpu().numpy() for v in (b,s,c))
                bb=extraer_bboxes_region(y[z]);g=[[v['clase_idx']]+v['bbox'] for v in bb]
                records.append({'case':cid,'z':int(z),'gt':g,'boxes':b.tolist(),'scores':s.tolist(),'labels':c.tolist()})
                gtclasses=[int(any(v['clase_idx']==i for v in bb)) for i in range(3)];truth.append(gtclasses);probs.extend(out['clases'].sigmoid().cpu().tolist())
                # Submuestra SAM predefinida por costo CPU, independiente de GT o desempeño.
                if z != int(np.rint(.5*(len(x)-1))):continue
                keep=s>=.25;boxes,scores,classes=b[keep],s[keep],c[keep]
                sem=gate_semantic(out['mascaras'][0].softmax(0).cpu().numpy(),boxes,classes)
                edge=out['bordes'][0].sigmoid().cpu().numpy()
                cores=out['interiores'][0].sigmoid().cpu().numpy()
                sam_sem=np.zeros((256,256),np.uint8);claimed=np.zeros((256,256),bool)
                if device.type=='cuda':torch.cuda.synchronize()
                start=time.perf_counter()
                if len(boxes):
                    rgb=np.repeat(np.rint(gray*255).astype(np.uint8)[...,None],3,axis=2);predictor.set_image(rgb)
                    for j in np.argsort(-scores):
                        masks,_,_=predictor.predict(box=boxes[j]*256,multimask_output=False)
                        mask=masks[0]&~claimed;sam_sem[mask]=int(classes[j])+1;claimed|=mask
                if device.type=='cuda':torch.cuda.synchronize()
                timings.append((time.perf_counter()-start)*1000)
                gtsem=semantic_labels(ids)
                for name,pred,be in [('propio',sem,edge),('sam',sam_sem,np.zeros_like(edge))]:
                    matrices[name]+=np.bincount((gtsem*4+pred).ravel(),minlength=16).reshape(4,4)
                    policy2d={**frozen_policy['instance_policy'],'min_volume_mm3':16.,'seed_mm3':64.} if name=='propio' else {'method':'componentes','min_volume_mm3':16.}
                    inst,mapping=reconstruir(pred[None],be[None],cores[None],(1,1,1),policy2d)
                    rows=match_instances(inst,ids[None],mapping)
                    instance_rows[name].extend({'case':cid,'z':int(z),**r} for r in rows)
                np.savez_compressed(a.output/f'{cid}_{z}.npz',image=gray,gt_instances=ids,own_semantic=sem,sam_semantic=sam_sem,boxes=boxes,labels=classes)
                print(f'Test {cid} z={z}: {len(boxes)} prompts SAM',flush=True)
            del x,y,img,label
    result={'detection_own':detection_metrics(records),'classification_own':classification_metrics(truth,probs),'semantic':{k:semantic_metrics(v) for k,v in matrices.items()},'instances_2d':{},'slices':len(records),'sam_comparison_slices':len(timings),'sam_mean_ms_including_skipped_empty':float(np.mean(timings))}
    for k,rows in instance_rows.items():
        gt=[r for r in rows if r['gt'] is not None]
        result['instances_2d'][k]={'gt_fragments_in_slices':len(gt),'dice_gt_macro_with_misses':float(np.mean([r['dice'] for r in gt])),'iou_gt_macro_with_misses':float(np.mean([r['iou'] for r in gt])),'dice_symmetric_macro_with_unmatched':float(np.mean([r['dice'] for r in rows])) if rows else None,'missed_gt':sum(r['pred'] is None for r in rows),'extra_pred':sum(r['gt'] is None for r in rows)}
        save(a.output/f'instancias_{k}.json',rows)
    save(a.output/'metricas.json',result);save(a.output/'detecciones.json',records)
    print(json.dumps(result,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
