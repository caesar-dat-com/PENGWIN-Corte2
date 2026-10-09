"""Comparación en val del experimento previo y la adaptación; no abre test."""
import sys,json,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from entrenar_pacientes import Samples,collate,decode,evaluate
from pengwin.models.detector import PelvisDetector
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.instances import semantic_labels,reconstruct_instances,match_instances
from pengwin.fragmentos_sesion2 import reconstruir_fragmentos
from pengwin.metrics_avance3 import gate_semantic


def main():
    p=argparse.ArgumentParser();p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(a.threads)
    out=ROOT/'salidas/sesion2';out.mkdir(exist_ok=True)
    data=Samples(ROOT/'salidas/revision_oct08/datos','val')
    loader=DataLoader(data,batch_size=8,collate_fn=collate)
    models={};results={};rows={};hashes={}
    for name,cls,folder in [('previo',PelvisDetector,'revision_oct08'),('sesion2',PelvisSesion2,'sesion2')]:
        path=ROOT/f'salidas/{folder}/entrenamiento/best.pth'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        model=cls(pretrained=False) if name=='previo' else cls()
        model.load_state_dict(ck['model']);model.eval();models[name]=model
        results[name],_=evaluate(model,loader,torch.device('cpu'));rows[name]=[]
        hashes[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'epoch':ck['epoch']}
    candidates={(seed,area):[] for seed in (4,16,64) for area in (0,4,16)}
    examples=[];chosen=json.loads((ROOT/'splits/splits.json').read_text())['val'][:3]
    with torch.inference_mode():
        for row in data.rows:
            cid,j=row['case'],row['index'];cached=data.cache[cid]
            image=cached['images'][j];ids=cached['instances'][j]
            panels={}
            for name,model in models.items():
                pred=model(torch.from_numpy(image)[None,None].repeat(1,3,1,1))
                b,s,c=[v.numpy() for v in decode(pred)]
                sem=gate_semantic(pred['mascaras'][0].softmax(0).numpy(),b,c)
                edge=pred['bordes'][0].sigmoid().numpy()
                if name=='sesion2':
                    cores=pred['interiores'][0].sigmoid().numpy()
                    inst,mapping=reconstruir_fragmentos(sem[None],edge[None],cores[None],(1,1,1),min_volume_mm3=0)
                    for (seed,area),rr in candidates.items():
                        ci,cm=reconstruir_fragmentos(sem[None],edge[None],cores[None],(1,1,1),min_volume_mm3=area,seed_min_volume_mm3=seed)
                        rr.extend(match_instances(ci,ids[None],cm))
                else:inst,mapping=reconstruct_instances(sem[None],edge[None],(1,1,1),min_volume_mm3=0)
                rows[name].extend({'case':cid,'z':row['z'],**r} for r in match_instances(inst,ids[None],mapping))
                panels[name]=(sem,inst[0])
            if cid in chosen and j==4:examples.append((cid,row['z'],image,semantic_labels(ids),panels))
    for name,rr in rows.items():
        gt=[r for r in rr if r['gt'] is not None]
        results[name]['instances_2d']={'gt':len(gt),'missed':sum(r['pred'] is None for r in rr),'extra':sum(r['gt'] is None for r in rr),
            'dice_gt_macro':float(np.mean([r['dice'] for r in gt])),
            'dice_symmetric':float(np.mean([r['dice'] for r in rr]))}
        (out/f'instancias_val_{name}.json').write_text(json.dumps(rr,indent=2),encoding='utf-8')
    document={'scope':'120 cortes de 15 pacientes val; ninguna evaluación test nueva. Instancias 2D no equivalen a fragmentos 3D.',
        'checkpoints':hashes,'metrics':results,'comparison':'Ambos desde cero, 6 épocas, mismo split/muestreo. Cambio conjunto: skips, supervisión profunda e interiores; no aísla el efecto individual de cada componente.'}
    (out/'comparacion_val.json').write_text(json.dumps(document,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    sensitivity=[]
    for (seed,area),rr in candidates.items():
        gt=[r for r in rr if r['gt'] is not None]
        sensitivity.append({'min_seed_pixels':seed,'min_instance_pixels':area,'dice_gt_macro':float(np.mean([r['dice'] for r in gt])),
            'dice_symmetric':float(np.mean([r['dice'] for r in rr])),'missed':sum(r['pred'] is None for r in rr),'extra':sum(r['gt'] is None for r in rr)})
    best=max(sensitivity,key=lambda d:d['dice_symmetric'])
    (out/'calibracion_instancias_2d.json').write_text(json.dumps({'scope':'Ajuste sobre val, no resultado independiente de prueba. Umbrales en píxeles 2D, no mm3. No modifican GT.','selection':'máximo Dice simétrico, penaliza extras y omisiones','candidates':sensitivity,'selected':best},indent=2),encoding='utf-8')
    cmap=ListedColormap(['black','#ee643d','#298bea','#14bda9'])
    fig,axs=plt.subplots(len(examples),4,figsize=(14,3.6*len(examples)),squeeze=False,layout='constrained')
    for i,(cid,z,image,gt,panels) in enumerate(examples):
        for j,(title,mask) in enumerate([('GT regiones',gt),('Decoder previo',panels['previo'][0]),('Decoder con skips',panels['sesion2'][0]),('Instancias 2D nuevas',panels['sesion2'][1])]):
            ax=axs[i,j];ax.imshow(image,cmap='gray',vmin=0,vmax=1)
            ax.imshow(np.ma.masked_equal(mask,0),cmap=cmap if j<3 else 'tab20',vmin=0,vmax=3 if j<3 else max(int(mask.max()),1),alpha=.6)
            ax.set_title(f'{cid} · z={z}\n{title}');ax.axis('off')
    fig.savefig(out/'comparacion_visual.png',dpi=160);plt.close(fig)
    print(json.dumps(document,ensure_ascii=True))


if __name__=='__main__':main()
