"""Exporta la separación seleccionada sin sobrescribir evidencia anterior."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,SimpleITK as sitk
from evaluar_instancias_cierre import evaluate

def main():
    report=json.loads((ROOT/'salidas/cajas_revision/separacion/comparacion.json').read_text(encoding='utf-8'))
    selected=report['selected']
    if selected is None:raise ValueError('Ninguna variante elegible')
    audit=json.loads((ROOT/'salidas/cajas_revision/cuerpo/auditoria.json').read_text(encoding='utf-8'))
    if any(c['gt_voxels_removed'] for c in audit['cases']):raise ValueError('Filtro corporal requiere revisión de conservación de GT')
    for cid in ('002','012','028'):
        src=ROOT/'salidas/cajas_revision/volumenes'/cid
        dest=ROOT/'salidas/fragmentos_revision/volumenes'/cid
        if dest.exists():raise ValueError('Salida existente')
        dest.mkdir(parents=True)
        for name in ('bordes.mha','interiores.mha','ct_ventana.mha','boxes.json','prediccion.json'):
            shutil.copy2(src/name,dest/name)
        body_path=ROOT/'salidas/cajas_revision/cuerpo'/f'{cid}_cuerpo.mha'
        body_image=sitk.ReadImage(str(body_path));body=sitk.GetArrayFromImage(body_image)>0
        ref=sitk.ReadImage(str(src/'semantica.mha'));sem=sitk.GetArrayFromImage(ref)
        assert ref.GetSize()==body_image.GetSize()
        for key in ('GetDirection','GetOrigin','GetSpacing'):
            assert np.allclose(getattr(ref,key)(),getattr(body_image,key)())
        sem[~body]=0
        obj=sitk.GetImageFromArray(sem);obj.CopyInformation(ref);sitk.WriteImage(obj,str(dest/'semantica.mha'),True)
        shutil.copy2(body_path,dest/'cuerpo.mha')
        metadata=json.loads((dest/'prediccion.json').read_text(encoding='utf-8'))
        metadata['body_policy']=audit['policy'];metadata['instance_policy']=selected['policy']
        (dest/'prediccion.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(cid,evaluate(cid,selected['policy'],True,'salidas/fragmentos_revision/volumenes'),flush=True)

if __name__=='__main__':main()
