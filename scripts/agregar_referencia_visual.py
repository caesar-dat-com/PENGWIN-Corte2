"""Superposición opcional de GT solo para inspección; no cambia predicciones."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import SimpleITK as sitk
from pengwin.dataset import extraer_bboxes_region

if __name__=='__main__':
    for p in sorted((ROOT/'dashboard/data').glob('*/cortes.json')):
        ids=sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT/'salidas/cierre/volumenes'/p.parent.name/'gt_evaluacion.mha')))
        rows=json.loads(p.read_text(encoding='utf-8'))
        for row in rows:
            boxes=extraer_bboxes_region(ids[row['z']]);row['reference_boxes']=[r['bbox'] for r in boxes];row['reference_labels']=[r['clase_idx'] for r in boxes]
        p.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    for name in ('index.html','app.js','style.css'):shutil.copy2(ROOT/'web'/name,ROOT/'dashboard'/name)
    print('Referencia visual agregada sin cambiar cajas predichas')
