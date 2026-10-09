"""Comparación visual en posiciones fijas de un volumen de validación."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,SimpleITK as sitk
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pengwin.io import resolver_volumen,normalizar_lps
from pengwin.instances import semantic_labels

def main():
    cid='002';p=ROOT/'salidas/cierre/volumenes'/cid
    ref=sitk.ReadImage(str(p/'semantica.mha'));current=sitk.GetArrayFromImage(ref)
    old=sitk.ReadImage(str(ROOT/'salidas/macro_sobel/volumenes'/cid/'semantica.mha'))
    assert old.GetSize()==ref.GetSize() and np.allclose(old.GetOrigin(),ref.GetOrigin()) and np.allclose(old.GetSpacing(),ref.GetSpacing()) and np.allclose(old.GetDirection(),ref.GetDirection())
    previous=sitk.GetArrayFromImage(old);gray=sitk.GetArrayFromImage(sitk.ReadImage(str(p/'ct_ventana.mha')))
    cfg=json.loads((ROOT/'config_datos.local.json').read_text(encoding='utf-8'));gt=normalizar_lps(sitk.ReadImage(str(resolver_volumen(Path(cfg['labels']),cid))))
    gt=semantic_labels(sitk.GetArrayFromImage(sitk.Resample(gt,ref,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)))
    indices=np.rint(np.array([.25,.5,.75])*(len(current)-1)).astype(int)
    cmap=ListedColormap(['black','#ef714a','#58a9ff','#36c7aa']);fig,axs=plt.subplots(3,3,figsize=(11,11),layout='constrained')
    for row,z in zip(axs,indices):
        for ax,(title,mask) in zip(row,[('Referencia',gt),('Anterior: macro + refinamiento',previous),('Actual: ajuste de contorno',current)]):
            ax.imshow(gray[z],cmap='gray',vmin=0,vmax=255);ax.imshow(np.ma.masked_equal(mask[z],0),cmap=cmap,vmin=0,vmax=3,alpha=.5);ax.set_aspect(ref.GetSpacing()[1]/ref.GetSpacing()[0]);ax.set_title(f'{title}\nCaso 002 · corte {z+1}/{len(current)}',fontsize=10);ax.axis('off')
    fig.suptitle('Validación · posiciones 25 %, 50 % y 75 %\nMáscara anatómica; no demuestra separación individual de fragmentos',fontsize=13)
    fig.savefig(ROOT/'salidas/cierre/comparacion_visual_val.png',dpi=140);plt.close(fig)
    print('Comparación visual de validación guardada')
if __name__=='__main__':main()
