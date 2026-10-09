"""Ablación controlada de pérdida macro/Sobel y vecindad, solo validación."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scipy import ndimage as ndi
from torch.utils.data import DataLoader
from entrenar_sesion2 import Samples,collate,decode,evaluate
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.refinamiento_macro import refinar_probabilidades,control_calidad
from pengwin.metrics_avance3 import gate_semantic
from pengwin.fragmentos_sesion2 import reconstruir_fragmentos
from pengwin.instances import match_instances

def save(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def metrics(cm):
    den=cm.sum(0)[1:]+cm.sum(1)[1:];d=2*np.diag(cm)[1:]/np.maximum(den,1)
    return {'dice':d.tolist(),'macro_dice':float(d.mean()),'confusion':cm.tolist()}

def main():
    torch.set_num_threads(4);out=ROOT/'salidas/macro_sobel';out.mkdir(exist_ok=True)
    dataset=Samples(ROOT/'salidas/revision_oct08/datos','val');loader=DataLoader(dataset,batch_size=8,collate_fn=collate)
    paths={'inicial':'salidas/sesion2/entrenamiento/best.pth','control':'salidas/macro_control/entrenamiento/best.pth','macro_sobel':'salidas/macro_sobel/entrenamiento/best.pth'}
    for path in paths.values():
        config=json.loads((ROOT/path).with_name('config.json').read_text())
        if config.get('status')!='completed':raise ValueError('Terminar el entrenamiento antes de comparar: '+path)
    results={};examples={};candidates=[]
    for name,path in paths.items():
        checkpoint=torch.load(ROOT/path,map_location='cpu',weights_only=False);model=PelvisSesion2();model.load_state_dict(checkpoint['model']);model.eval()
        base,_=evaluate(model,loader,torch.device('cpu'));results[name]={'base':base,'sha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),'epoch':checkpoint['epoch'],'refinement':{}}
        cms={s:np.zeros((4,4),np.int64) for s in (0.,.25,.5)};ungated=np.zeros((4,4),np.int64);rows={s:[] for s in cms};quality=[]
        contour={s:np.zeros((3,4),float) for s in cms}
        with torch.inference_mode():
            for k in range(len(dataset)):
                sample=dataset[k];pred=model(sample['x'][None]);prob=pred['mascaras'][0].softmax(0).numpy();im=sample['x'][0].numpy();gt=sample['semantic'].numpy()
                b,score,c=[v.numpy() for v in decode(pred)];edge=pred['bordes'][0].sigmoid().numpy();core=pred['interiores'][0].sigmoid().numpy()
                ungated+=np.bincount((gt*4+prob.argmax(0)).ravel(),minlength=16).reshape(4,4)
                cid,z=sample['case'],sample['z'];idx=dataset.rows[k]['index'];ids=dataset.cache[cid]['instances'][idx]
                panels={}
                for strength in cms:
                    refined=refinar_probabilidades(prob,im,strength=strength);sem=gate_semantic(refined,b,c)
                    cms[strength]+=np.bincount((gt*4+sem).ravel(),minlength=16).reshape(4,4)
                    # Mismos umbrales 2D elegidos en la revisión anterior, sin volver a ajustarlos.
                    inst,mapping=reconstruir_fragmentos(sem[None],edge[None],core[None],(1,1,1),min_volume_mm3=16,seed_min_volume_mm3=64)
                    rows[strength].extend(match_instances(inst,ids[None],mapping))
                    for r in (1,2,3):
                        pm=sem==r;tm=gt==r;pe=pm&~ndi.binary_erosion(pm);te=tm&~ndi.binary_erosion(tm)
                        contour[strength][r-1]+= [np.count_nonzero(pe&ndi.binary_dilation(te,iterations=2)),pe.sum(),np.count_nonzero(te&ndi.binary_dilation(pe,iterations=2)),te.sum()]
                    if cid in json.loads((ROOT/'splits/splits.json').read_text())['val'][:3] and idx==4:panels[str(strength)]=sem
                quality.append({'case':cid,'z':z,**control_calidad(gate_semantic(prob,b,c),prob)})
                if panels:examples.setdefault((cid,z),{'image':im,'gt':gt})[name]=panels
        results[name]['ungated_semantic']=metrics(ungated)
        for s,cm in cms.items():
            sem=metrics(cm);rr=rows[s];gg=[r for r in rr if r['gt'] is not None];ct=contour[s]
            precision=ct[:,0]/np.maximum(ct[:,1],1);recall=ct[:,2]/np.maximum(ct[:,3],1)
            sem['boundary_f1_tolerance_2_pixels']=(2*precision*recall/np.maximum(precision+recall,1e-8)).tolist()
            sem['instances_2d']={'gt':len(gg),'missed':sum(r['pred'] is None for r in rr),'extra':sum(r['gt'] is None for r in rr),'dice_symmetric':float(np.mean([r['dice'] for r in rr])),'dice_gt':float(np.mean([r['dice'] for r in gg]))}
            results[name]['refinement'][str(s)]=sem
            candidates.append({'model':name,'checkpoint':path,'strength':s,'score':.75*sem['macro_dice']+.25*min(sem['dice']),'macro_dice':sem['macro_dice'],'dice':sem['dice']})
        save(out/f'calidad_val_{name}.json',quality)
        print(name,results[name]['refinement']['0.0'],flush=True)
    # No sacrificar otra región ni la detección respecto al modelo que usa el equipo.
    reference=results['inicial']['refinement']['0.0']['dice']
    reference_ap=results['inicial']['base']['detection']['mAP50']
    for candidate in candidates:
        candidate['eligible']=bool(all(d+1e-6>=r for d,r in zip(candidate['dice'],reference)) and results[candidate['model']]['base']['detection']['mAP50']+1e-6>=reference_ap)
    # Incluye identidad y checkpoint previo: no obliga a aceptar un experimento peor.
    selected=max((c for c in candidates if c['eligible']),key=lambda x:x['score'])
    selected.update(checkpoint_sha256=results[selected['model']]['sha256'],scope='Seleccion y ajuste sobre val; no es evaluacion independiente test',seed_min_mm3_3d=50.,instance_min_mm3_3d=20.)
    save(out/'comparacion.json',{'scope':'120 cortes,15 pacientes val; control y candidato parten del mismo checkpoint y entrenan 6 epocas adicionales con igual semilla/lr/lotes; paquete macro+Sobel, sin aislar sus efectos entre si','metrics':results,'candidates':candidates,'selected':selected})
    save(out/'seleccion.json',selected)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap=ListedColormap(['black','#ef6a40','#328eea','#11b8a4']);fig,axes=plt.subplots(len(examples),5,figsize=(16,3.8*len(examples)),squeeze=False,layout='constrained')
    for axrow,((cid,z),ex) in zip(axes,examples.items()):
        entries=[('Referencia',ex['gt']),('Inicial',ex['inicial']['0.0']),('Más entrenamiento',ex['control']['0.0']),('Macro + Sobel',ex['macro_sobel']['0.0']),('Seleccionado',ex[selected['model']][str(selected['strength'])])]
        for ax,(title,mask) in zip(axrow,entries):
            ax.imshow(ex['image'],cmap='gray',vmin=0,vmax=1);ax.imshow(np.ma.masked_equal(mask,0),cmap=cmap,vmin=0,vmax=3,alpha=.55);ax.set_title(f'{cid}, z={z}\n{title}');ax.axis('off')
    fig.savefig(out/'comparacion.png',dpi=150);plt.close(fig)
    print('SELECTED',selected,flush=True)
if __name__=='__main__':main()
