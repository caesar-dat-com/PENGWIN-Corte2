"""Mide predicciones externas y conservación de GT; GT nunca crea el filtro."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import SimpleITK as sitk
from pengwin.io import normalizar_lps,resolver_volumen
from pengwin.mascara_corporal import mascara_corporal

def main():
    cfg=json.loads((ROOT/'config_datos.local.json').read_text(encoding='utf-8'))
    dest=ROOT/'salidas/cajas_revision/cuerpo';dest.mkdir(exist_ok=True)
    results=[]
    for cid in ('002','012','028'):
        src=ROOT/'salidas/cajas_revision/volumenes'/cid
        ref=sitk.ReadImage(str(src/'semantica.mha'));sem=sitk.GetArrayFromImage(ref)
        image=normalizar_lps(sitk.ReadImage(str(resolver_volumen(Path(cfg['images']),cid))))
        ct=sitk.GetArrayFromImage(sitk.Resample(image,ref,sitk.Transform(),sitk.sitkLinear,-1024.,sitk.sitkFloat32))
        del image
        body=mascara_corporal(ct,ref.GetSpacing()[::-1]);del ct
        obj=sitk.GetImageFromArray(body.astype(np.uint8));obj.CopyInformation(ref);sitk.WriteImage(obj,str(dest/f'{cid}_cuerpo.mha'),True)
        gt=sitk.GetArrayFromImage(sitk.ReadImage(str(src/'gt_evaluacion.mha')))
        inst=sitk.GetArrayFromImage(sitk.ReadImage(str(src/'instancias.mha')))
        rows=[]
        for i in np.unique(inst):
            if i:rows.append({'id':int(i),'voxels':int((inst==i).sum()),'outside':int(((inst==i)&~body).sum())})
        result={'case':cid,'predicted_voxels_removed':int(((sem>0)&~body).sum()),'gt_voxels_removed':int(((gt>0)&~body).sum()),'gt_total_voxels':int((gt>0).sum()),'instances':rows}
        results.append(result);print(json.dumps(result),flush=True)
    (dest/'auditoria.json').write_text(json.dumps({'policy':{'threshold_hu':-500,'opening_mm':2,'margin_mm':3},'cases':results},indent=2),encoding='utf-8')

if __name__=='__main__':main()
