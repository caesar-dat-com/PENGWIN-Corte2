"""Visualiza conteos reales; no clasifica anotaciones como ruido."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'salidas/revision_oct08'
    audit=json.loads((out/'auditoria.json').read_text(encoding='utf-8'));rows=audit['rows']
    fig,ax=plt.subplots(figsize=(12,6))
    fig.subplots_adjust(bottom=.19,top=.83,left=.10,right=.97)
    bars=ax.bar([str(r['umbral_area_px']) for r in rows],[r['cajas_omitidas'] for r in rows],color='#187b9c');ax.bar_label(bars,padding=4)
    ax.set(title='Anotaciones que se excluirían según el área mínima\n100 máscaras · 62.738 regiones por corte',xlabel='Umbral de área en píxeles (se excluye área < umbral)',ylabel='Cajas de regiones anotadas excluidas',ylim=(0,960))
    ax.spines[['top','right']].set_visible(False)
    fig.text(.5,.03,'Son anotaciones GT: su tamaño pequeño no demuestra que sean ruido. Umbral operativo: 0.',ha='center',fontsize=10)
    fig.savefig(out/'auditoria_areas.png',dpi=170,bbox_inches='tight');plt.close(fig)
    print(out/'auditoria_areas.png')
if __name__=='__main__':main()
