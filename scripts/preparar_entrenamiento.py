"""Muestreo uniforme de train/val con IDs originales, sin abrir test."""
import sys,json,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2,SimpleITK as sitk
from pengwin.io import rutas_dataset,resolver_volumen,verificar_alineacion,ventana_hu
from pengwin.dataset import extraer_bboxes_region

def main():
    p=argparse.ArgumentParser();p.add_argument('--cortes',type=int,default=8);args=p.parse_args()
    out=ROOT/'salidas/revision_oct08/datos';out.mkdir(parents=True,exist_ok=True)
    images,labels=rutas_dataset();spfile=ROOT/'splits/splits.json';splits=json.loads(spfile.read_text())
    ids=[cid for s in ('train','val','test') for cid in splits[s]]
    if len(ids)!=len(set(ids)):raise ValueError('Splits solapados')
    proto={'cortes_por_paciente':args.cortes,'split_sha256':hashlib.sha256(spfile.read_bytes()).hexdigest(),'resolucion':256,'muestreo':'uniforme 0..100%, independiente de anotaciones, incluye negativos','images':str(images),'labels':str(labels)}
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
                zz=np.unique(np.rint(np.linspace(0,len(x)-1,args.cortes)).astype(int))
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
