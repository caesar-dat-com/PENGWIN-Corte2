"""Muestreo de cortes train/val con IDs originales, sin abrir test.

--modo uniforme: N cortes equiespaciados por paciente (avance oct-08).
--modo hueso: todos los cortes con hueso (cada --paso) + una fracción --vacios
de cortes sin hueso. Con 8 cortes/paciente el sacro solo aparecía en 52/120
cortes de val y el decoder colapsaba el coxal izquierdo."""
import sys,json,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2,SimpleITK as sitk
from pengwin.io import rutas_dataset,resolver_volumen,verificar_alineacion,ventana_hu
from pengwin.dataset import extraer_bboxes_region

def main():
    p=argparse.ArgumentParser();p.add_argument('--cortes',type=int,default=8)
    p.add_argument('--modo',choices=('uniforme','hueso'),default='uniforme');p.add_argument('--paso',type=int,default=1)
    p.add_argument('--vacios',type=float,default=.15);p.add_argument('--salida',type=Path,default=ROOT/'salidas/revision_oct08/datos');args=p.parse_args()
    out=args.salida;out.mkdir(parents=True,exist_ok=True)
    images,labels=rutas_dataset();spfile=ROOT/'splits/splits.json';splits=json.loads(spfile.read_text())
    ids=[cid for s in ('train','val','test') for cid in splits[s]]
    if len(ids)!=len(set(ids)):raise ValueError('Splits solapados')
    muestreo='uniforme 0..100%, independiente de anotaciones, incluye negativos' if args.modo=='uniforme' else f'todos los cortes con hueso cada {args.paso} + {args.vacios:.0%} de cortes vacíos (semilla 42)'
    proto={'cortes_por_paciente':args.cortes if args.modo=='uniforme' else None,'modo':args.modo,'split_sha256':hashlib.sha256(spfile.read_bytes()).hexdigest(),'resolucion':256,'muestreo':muestreo,'images':str(images),'labels':str(labels)}
    if (out/'protocolo.json').exists() and json.loads((out/'protocolo.json').read_text())!=proto:raise ValueError('Preparación incompatible')
    (out/'protocolo.json').write_text(json.dumps(proto,indent=2),encoding='utf-8');rows=[]
    for split in ('train','val'):
        for cid in splits[split]:
            dest=out/f'{cid}.npz'
            if not dest.exists():
                image=sitk.DICOMOrient(sitk.ReadImage(str(resolver_volumen(images,cid))),'LPS')
                label=sitk.DICOMOrient(sitk.ReadImage(str(resolver_volumen(labels,cid))),'LPS');verificar_alineacion(image,label,cid)
                x,y=sitk.GetArrayFromImage(image),sitk.GetArrayFromImage(label)
                if not np.isin(y,np.arange(31)).all():raise ValueError('Etiquetas inválidas')
                if args.modo=='uniforme':zz=np.unique(np.rint(np.linspace(0,len(x)-1,args.cortes)).astype(int))
                else:
                    con=np.flatnonzero(y.reshape(len(y),-1).any(1));sin=np.setdiff1d(np.arange(len(y)),con)
                    rng=np.random.default_rng(int(cid)+42);k=min(len(sin),int(round(len(con)/args.paso*args.vacios)))
                    zz=np.sort(np.concatenate([con[::args.paso],rng.choice(sin,k,replace=False) if k else np.array([],int)])).astype(int)
                xx=np.stack([cv2.resize(ventana_hu(x[z]),(256,256),interpolation=cv2.INTER_AREA) for z in zz])
                yy=np.stack([cv2.resize(y[z].astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST) for z in zz])
                boxes=np.full((len(zz),3,5),-1.,np.float32)
                for i,z in enumerate(zz):
                    for j,b in enumerate(extraer_bboxes_region(y[z],min_pixeles=0)):boxes[i,j]=[b['clase_idx']]+b['bbox']
                tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,images=xx,instances=yy,boxes=boxes,z=zz);tmp.replace(dest)
                del image,label,x,y
            with np.load(dest) as data:
                rows.extend({'case':cid,'split':split,'index':i,'z':int(z)} for i,z in enumerate(data['z']))
            print(f'{split} {cid}',flush=True)
    (out/'manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':main()
