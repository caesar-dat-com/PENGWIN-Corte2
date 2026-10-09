"""Verifica coherencia de cajas, máscaras y geometría de los volúmenes revisados."""
import json,argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results-root',default='salidas/cajas_revision/volumenes');args=parser.parse_args()
    cases=[]
    for cid in ('002','012','028'):
        folder=ROOT/args.results_root/cid
        rows=json.loads((folder/'boxes.json').read_text(encoding='utf-8'))
        image=sitk.ReadImage(str(folder/'semantica.mha'))
        sem=sitk.GetArrayFromImage(image)
        assert np.allclose(image.GetDirection(),np.eye(3).ravel())
        assert len(rows)==len(sem)
        for name in ('instancias','gt_evaluacion','ct_ventana'):
            other=sitk.ReadImage(str(folder/f'{name}.mha'))
            assert other.GetSize()==image.GetSize()
            for key in ('GetSpacing','GetOrigin','GetDirection'):
                assert np.allclose(getattr(other,key)(),getattr(image,key)())
        for z,row in enumerate(rows):
            assert len(row['labels'])==len(set(row['labels']))
            for cls in range(3):
                allowed=np.zeros(sem[z].shape,bool)
                for box,label in zip(row['boxes'],row['labels']):
                    assert 0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1
                    if cls==label:
                        x1,y1=np.floor(np.array(box[:2])*256).astype(int)
                        x2,y2=np.ceil(np.array(box[2:])*256).astype(int)
                        allowed[y1:y2,x1:x2]=True
                assert not np.any((sem[z]==cls+1)&~allowed)
        summary=json.loads((folder/'resumen.json').read_text(encoding='utf-8'))
        cases.append({'case':cid,'slices':len(rows),'duplicate_boxes':0,'geometry':'LPS aligned','mask_within_boxes':True,'summary':summary})
    result={'status':'passed','scope':'3 validation volumes; no independent test','cases':cases}
    (ROOT/args.results_root).parent.joinpath('verificacion_volumenes.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
