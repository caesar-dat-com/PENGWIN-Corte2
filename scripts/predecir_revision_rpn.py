"""Predicciones contiguas reutilizables: GT se carga después y solo al evaluar."""
import sys,json,time,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,SimpleITK as sitk,cv2
from pengwin.io import resolver_volumen,normalizar_lps,ventana_hu
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.metrics_avance3 import gate_semantic
from entrenar_sesion2 import decode
from pengwin.models.rpn_grid import RPNGridHead,decode_proposals
from pengwin.propuestas_regionales import combinar_regiones
from evaluar_instancias_cierre import evaluate
from inferir_macro_3d import reference_grid

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',required=True);p.add_argument('--include-test',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg=json.loads((ROOT/'config_datos.local.json').read_text(encoding='utf-8'));splits=json.loads((ROOT/'splits/splits.json').read_text(encoding='utf-8'))
    chosen=json.loads((ROOT/'salidas/cierre/seleccion_modelo.json').read_text(encoding='utf-8'));checkpoint=ROOT/chosen['checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=chosen['sha256']:raise ValueError('Checkpoint cambió')
    if any(c in splits['test'] for c in a.cases):
        if not a.include_test or not (ROOT/'salidas/cierre/protocolo_final.json').exists():raise ValueError('Congelar protocolo antes de abrir test')
    model=PelvisSesion2();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.to(device).eval()
    proposal_path=ROOT/'salidas/cajas_revision/rpn_grid/best_diagnostic.pth'
    proposal=torch.load(proposal_path,map_location=device,weights_only=False)
    head=RPNGridHead(proposal['config']['anchor_sizes']).to(device);head.load_state_dict(proposal['head']);head.eval()
    policy=json.loads((ROOT/'salidas/cajas_revision/fusion.json').read_text(encoding='utf-8'))['selected']
    instance_policy=json.loads((ROOT/'salidas/cierre/protocolo_final.json').read_text(encoding='utf-8'))['instance_policy']
    if any(c not in splits['val'] for c in a.cases):raise ValueError('Revision limitada a validacion')
    with torch.inference_mode():
        for _ in range(3):model(torch.zeros(1,3,256,256,device=device))
        for cid in a.cases:
            if not any(cid in splits[k] for k in ('train','val','test')):raise ValueError('Caso fuera del split')
            dest=ROOT/'salidas/cajas_revision/volumenes'/cid
            if dest.exists():raise ValueError('Salida existente: '+str(dest))
            image=normalizar_lps(sitk.ReadImage(str(resolver_volumen(Path(cfg['images']),cid))));x=sitk.GetArrayFromImage(image);ref=reference_grid(image)
            sem=np.zeros((len(x),256,256),np.uint8);edge=np.zeros_like(sem,np.float32);core=np.zeros_like(edge);gray=np.zeros_like(sem);boxes=[];times=[]
            for z in range(len(x)):
                im=cv2.resize(ventana_hu(x[z]),(256,256),interpolation=cv2.INTER_AREA);gray[z]=np.rint(im*255).astype(np.uint8)
                inp=torch.from_numpy(im)[None,None].repeat(1,3,1,1).to(device)
                if device.type=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();out=model(inp);ob,os,oc=[v.cpu().numpy() for v in decode(out,conf=.05)]
                nb,ns,nc=[v.cpu().numpy() for v in decode_proposals(head,head(out['features']),conf=.05,presence=out['clases'][0].sigmoid())]
                bb,ss,cc,sources=combinar_regiones(ob,os,oc,nb,ns,nc,new_conf=policy['new_conf'],old_conf=policy['old_conf'],padding=policy['padding'])
                sem[z]=gate_semantic(out['mascaras'][0].softmax(0).cpu().numpy(),bb,cc);edge[z]=out['bordes'][0].sigmoid().cpu().numpy();core[z]=out['interiores'][0].sigmoid().cpu().numpy()
                if device.type=='cuda':torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000);boxes.append({'z':z,'boxes':bb.tolist(),'scores':ss.tolist(),'labels':cc.tolist(),'sources':sources})
            dest.mkdir(parents=True)
            for name,arr in [('semantica',sem),('bordes',edge),('interiores',core),('ct_ventana',gray)]:
                obj=sitk.GetImageFromArray(arr);obj.CopyInformation(ref);sitk.WriteImage(obj,str(dest/f'{name}.mha'),True)
            save(dest/'boxes.json',boxes)
            save(dest/'prediccion.json',{'detector':'RPN grid con respaldo; revision val','proposal_sha256':hashlib.sha256(proposal_path.read_bytes()).hexdigest(),'fusion_policy':{k:policy[k] for k in ('new_conf','old_conf','padding')},'case':cid,'split':next(k for k in ('train','val','test') if cid in splits[k]),'sha256':chosen['sha256'],'slices':len(x),'device':str(device),'latency_mean_ms':float(np.mean(times)),'latency_p50_ms':float(np.median(times)),'latency_p95_ms':float(np.percentile(times,95)),'latency_scope':'forward, NMS, restriccion de mascara por bbox y copia CPU; 3 warmups; excluye IO, resize e instancias3D','spacing_xyz':ref.GetSpacing(),'origin':ref.GetOrigin(),'direction':ref.GetDirection()})
            evaluate(cid,instance_policy,True,results_root='salidas/cajas_revision/volumenes')
            print('VOLUMEN',cid,len(x),'cortes',flush=True)
if __name__=='__main__':main()
