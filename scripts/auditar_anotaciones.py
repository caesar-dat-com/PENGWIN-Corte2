"""Conteo REAL del área anotada por región y corte, sin llamarla ruido."""
import sys,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import SimpleITK as sitk
from pengwin.io import rutas_dataset,resolver_volumen

def main():
    _,labels=rutas_dataset();out=ROOT/'salidas/revision_oct08';out.mkdir(parents=True,exist_ok=True)
    thresholds=[0,5,10,15,20,30,50];counts={n:0 for n in thresholds};total=0;cases=[]
    splits=json.loads((ROOT/'splits/splits.json').read_text())
    ids=sorted(cid for s in ('train','val','test') for cid in splits[s])
    for cid in ids:
        image=sitk.DICOMOrient(sitk.ReadImage(str(resolver_volumen(labels,cid))),'LPS')
        a=sitk.GetArrayFromImage(image)
        if not np.isin(a,np.arange(31)).all():raise ValueError(cid+': etiquetas inválidas')
        row={'caso':cid,'apariciones':0,'area_menor_15':0}
        for region in range(3):
            area=((a>=region*10+1)&(a<=region*10+10)).sum(axis=(1,2))
            positive=area[area>0];total+=len(positive);row['apariciones']+=len(positive)
            row['area_menor_15']+=int((positive<15).sum())
            for n in thresholds:counts[n]+=int((positive<n).sum())
        cases.append(row);print(f'Auditado {cid}: {row}',flush=True)
    rows=[{'umbral_area_px':n,'regiones_corte_anotadas':total,'cajas_omitidas':counts[n],
           'cajas_conservadas':total-counts[n],'porcentaje_omitido':100*counts[n]/total} for n in thresholds]
    for name,data in [('auditoria_umbral.csv',rows),('auditoria_por_caso.csv',cases)]:
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=data[0]);writer.writeheader();writer.writerows(data)
    (out/'auditoria.json').write_text(json.dumps({'casos':len(ids),'rows':rows,'nota':'Conteo de regiones anotadas por corte, NO componentes conexas ni ruido confirmado. Umbral operativo por defecto=0, sin descartar GT. No se modificaron las máscaras.'},indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()
