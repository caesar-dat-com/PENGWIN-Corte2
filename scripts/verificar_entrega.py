"""Verifica consistencia de artefactos de la ejecución, sin entrenar ni ajustar."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np,SimpleITK as sitk,cv2

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    out=ROOT/'salidas/cierre';frozen=read(out/'protocolo_final.json')
    assert hashlib.sha256((ROOT/frozen['checkpoint']).read_bytes()).hexdigest()==frozen['checkpoint_sha256']
    splits=read(ROOT/'splits/splits.json');sets=[set(splits[k]) for k in ('train','val','test')]
    assert [len(x) for x in sets]==[70,15,15]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    catalog=read(ROOT/'dashboard/data/catalogo.json');assert len(catalog)==6
    counts={}
    for c in catalog:
        p=out/'volumenes'/c['id'];meta=read(p/'prediccion.json')
        assert meta['sha256']==frozen['checkpoint_sha256']
        images=[sitk.ReadImage(str(p/n)) for n in ('semantica.mha','instancias.mha','gt_evaluacion.mha')]
        ref=images[0]
        for im in images[1:]:
            assert im.GetSize()==ref.GetSize()
            assert np.allclose(im.GetSpacing(),ref.GetSpacing()) and np.allclose(im.GetOrigin(),ref.GetOrigin()) and np.allclose(im.GetDirection(),ref.GetDirection())
        rows=read(ROOT/f"dashboard/data/{c['id']}/cortes.json")
        assert len(rows)==meta['slices']==ref.GetSize()[2]
        for z,row in enumerate(rows):assert row['z']==z and (ROOT/'dashboard'/row['image']).is_file()
        for name in ('malla.html','mip.html'):
            html=(ROOT/f"dashboard/data/{c['id']}"/name).read_text(encoding='utf-8')
            assert '../../assets/plotly.min.js' in html and 'cdn.plot.ly' not in html
        comparisons=read(p/'comparacion_distancias.json')
        valid=[r for r in comparisons if r['valid_pair']]
        assert len(valid)==c['summary']['distance_valid_pairs']
        assert all(r['absolute_error_mm'] is None for r in comparisons if not r['valid_pair'])
        counts[c['id']]=len(rows)
    video=read(out/'video.json');cap=cv2.VideoCapture(str(ROOT/video['file']))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==video['frames'];cap.release()
    for name in ('INFORME_CIERRE.md','MODEL_CARD_FINAL.md','GUION_CIERRE.md'):assert (ROOT/'doc'/name).is_file()
    result={'status':'passed','checkpoint_sha256':frozen['checkpoint_sha256'],'patient_split':[70,15,15],'dashboard_contiguous_slices':counts,'video_frames':video['frames'],'scope':'Integridad de artefactos, geometría, hash de modelo, separación de pacientes y referencias locales. No demuestra calidad clínica ni metas métricas.'}
    (out/'verificacion_entrega.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
