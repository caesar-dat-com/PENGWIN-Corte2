"""Audita solo headers de los 100 casos; no usa test para ajustar el modelo."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import SimpleITK as sitk
from pengwin.io import rutas_dataset,resolver_volumen

def main():
    images,labels=rutas_dataset();splits=json.loads((ROOT/'splits/splits.json').read_text());rows=[]
    for split in ('train','val','test'):
        cases=splits[split]
        for cid in cases:
            meta=[]
            for folder in (images,labels):
                r=sitk.ImageFileReader();r.SetFileName(str(resolver_volumen(folder,cid)));r.ReadImageInformation()
                meta.append({'size':r.GetSize(),'spacing':r.GetSpacing(),'origin':r.GetOrigin(),'direction':r.GetDirection(),
                             'orientation':sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(r.GetDirection())})
            a,b=meta;aligned=all(np.allclose(a[k],b[k],atol=1e-3,rtol=0) for k in ('size','spacing','origin','direction'))
            d=np.array(a['direction']).reshape(3,3)
            valid=bool(np.isfinite(d).all() and np.allclose(d.T@d,np.eye(3),atol=1e-5,rtol=0) and min(a['spacing'])>0)
            rows.append({'case':cid,'split':split,'image':a,'label':b,'aligned_raw':aligned,'orthogonal_positive_spacing':valid})
    out=ROOT/'salidas/macro_sobel';out.mkdir(parents=True,exist_ok=True)
    result={'scope':'100 headers, sin voxeles test ni metricas test; LPS ya se usaba antes de este cambio',
            'all_aligned':all(r['aligned_raw'] for r in rows),'all_valid':all(r['orthogonal_positive_spacing'] for r in rows),'cases':rows}
    (out/'geometria.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
if __name__=='__main__':main()
