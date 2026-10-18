"""Gráficas reproducibles de las anotaciones originales; no son métricas del modelo."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'salidas'
def read(name):
    with (OUT/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))
C=read('eda_casos.csv');F=read('eda_fragmentos.csv');H=read('eda_hu_hueso.csv')
splits=json.loads((ROOT/'splits/splits.json').read_text())
cfg=json.loads((OUT/'fix_huesos_v1/config.json').read_text())
regions=['SA','LI','RI'];names=['Sacro','Coxal izquierdo','Coxal derecho'];colors=['#dd734a','#3e83be','#259f94']
non=[r for r in F if r['es_principal']=='False']
assert len({c['caso'] for c in C})==len(C)==100
assert sum(int(c['n_frag_total']) for c in C)==len(F)
assert set(c['caso'] for c in C)==set(splits['train']+splits['val']+splits['test'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#24394c','text.color':'#17334b','axes.titlesize':15,'svg.fonttype':'none'})
def fig(title,subtitle,n=2):
    f,ax=plt.subplots(1,n,figsize=(16,9),squeeze=False)
    f.suptitle(title,x=.07,y=.96,ha='left',fontsize=24,fontweight='bold')
    f.text(.07,.905,subtitle,fontsize=12,color='#526477')
    f.subplots_adjust(left=.08,right=.96,bottom=.17,top=.79,wspace=.3)
    return f,ax[0]
def save(f,name,source='eda_casos.csv / eda_fragmentos.csv'):
    f.text(.07,.055,'Fuente: '+source+' · Anotaciones de referencia (GT), no predicciones.',fontsize=10,color='#526477')
    for ext in ('png','svg'):f.savefig(OUT/(name+'.'+ext),dpi=180,facecolor='white')
    plt.close(f)
def bars(ax,labels,values,cs=colors):
    b=ax.bar(labels,values,color=cs);ax.bar_label(b,padding=5,fmt='%g');ax.set_ylim(0,max(values)*1.18 if max(values)>0 else 1);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)

f,a=fig('PENGWIN: panorama del conjunto de datos','100 pacientes · 3 regiones anatómicas · análisis del volumen completo',3)
bars(a[0],['Train','Validación','Prueba'],[len(splits[k]) for k in ['train','val','test']]);a[0].set_title('Pacientes por conjunto');a[0].set_ylabel('Pacientes')
total=sum(int(c['nz']) for c in C);bone=sum(int(c['cortes_con_hueso']) for c in C)
bars(a[1],['Con hueso\netiquetado','Sin hueso\netiquetado'],[bone,total-bone],['#259f94','#9daebb']);a[1].set_title(f'{total:,} cortes en total'.replace(',','.'));a[1].set_ylabel('Cortes')
bars(a[2],['Principales','No principales'],[len(F)-len(non),len(non)],['#3e83be','#dd734a']);a[2].set_title(f'{len(F)} instancias óseas');a[2].set_ylabel('Instancias')
save(f,'eda_01_panorama','eda_casos.csv / eda_fragmentos.csv / splits.json')

f,a=fig('Cortes disponibles y cortes seleccionados','Los cortes usados en entrenamiento no equivalen al total de cortes del dataset.')
keys=['train','val','test'];tot=[sum(int(c['nz']) for c in C if c['caso'] in splits[k]) for k in keys]
bars(a[0],['Train','Validación','Prueba'],tot);a[0].set_title('Cortes disponibles por conjunto');a[0].set_ylabel('Cortes completos')
bars(a[1],['Entrenamiento','Validación'],[cfg['train_slices'],cfg['val_slices']],colors[:2]);a[1].set_title('Selección del entrenamiento GPU');a[1].set_ylabel('Cortes seleccionados')
f.text(.55,.11,'Muestreo: cortes con hueso cada 2 + cortes vacíos.\nBatch registrado: 32 · 30 épocas.',fontsize=11)
save(f,'eda_02_cortes_conjuntos','eda_casos.csv / splits.json / fix_huesos_v1/config.json')

f,a=fig('Variación del número de cortes entre pacientes','Cada barra representa un paciente; el color distingue cortes con y sin hueso etiquetado.',1)
ordered=sorted(C,key=lambda c:int(c['nz']));x=np.arange(len(C));b=np.array([int(c['cortes_con_hueso']) for c in ordered]);t=np.array([int(c['nz']) for c in ordered])
a[0].bar(x,b,color='#259f94',label='Con hueso etiquetado');a[0].bar(x,t-b,bottom=b,color='#becad4',label='Sin hueso etiquetado');a[0].set_xticks(x[::5],[c['caso'] for c in ordered[::5]],rotation=60);a[0].set_xlabel('ID del paciente, ordenado por cantidad de cortes');a[0].set_ylabel('Cortes');a[0].legend();save(f,'eda_03_cortes_por_paciente','eda_casos.csv')

f,a=fig('Distribución anatómica de los fragmentos','Se incluyen los fragmentos principales en el conteo de instancias por región.')
counts=[sum(r['region']==k for r in F) for k in regions];fract=[sum(int(c['n_frag_'+k])>1 for c in C) for k in regions]
bars(a[0],names,counts);a[0].set_title('Instancias etiquetadas por región');a[0].set_ylabel('Instancias')
bars(a[1],names,fract);a[1].set_title('Pacientes con más de un fragmento');a[1].set_ylabel('Pacientes (de 100)');save(f,'eda_04_fragmentos_region')

f,a=fig('Complejidad de la fragmentación por paciente','Más de una instancia en una región se contabiliza como región fragmentada.')
vals=[int(c['n_frag_total']) for c in C];a[0].hist(vals,bins=np.arange(min(vals)-.5,max(vals)+1.5),color='#3e83be',rwidth=.8);a[0].set_xlabel('Instancias totales por paciente');a[0].set_ylabel('Pacientes');a[0].set_title('Incluye los tres principales')
bars(a[1],['0','1','2','3'],[sum(int(c['regiones_fracturadas'])==i for c in C) for i in range(4)],['#9daebb','#3e83be','#259f94','#dd734a']);a[1].set_title('Regiones fragmentadas por paciente');a[1].set_xlabel('Número de regiones');a[1].set_ylabel('Pacientes');save(f,'eda_05_complejidad','eda_casos.csv')

f,a=fig('Tamaño de los fragmentos no principales',f'{len(non)} fragmentos · el volumen se expresa en mililitros · escala logarítmica.')
data=[np.array([float(r['vol_ml']) for r in non if r['region']==k]) for k in regions]
bp=a[0].boxplot(data,tick_labels=names,patch_artist=True)
for box,color in zip(bp['boxes'],colors):box.set_facecolor(color)
a[0].set_yscale('log');a[0].set_ylabel('Volumen (mL, escala log)');a[0].set_title('Distribución por región')
for k,col,label in zip(regions,colors,names):
    rows=[r for r in non if r['region']==k];a[1].scatter([float(r['vol_ml']) for r in rows],[float(r['dist_mm_gt']) for r in rows],s=24,color=col,alpha=.65,label=label)
a[1].set_xscale('log');a[1].set_xlabel('Volumen (mL, escala log)');a[1].set_ylabel('Distancia EDT (mm)');a[1].legend();a[1].set_title('Volumen y separación registrada');save(f,'eda_06_volumen_fragmentos','eda_fragmentos.csv')

f,a=fig('Contacto y separación de fragmentos','La vecindad 26 y la distancia EDT describen propiedades distintas de la cuadrícula.')
contact=sum(r['en_contacto']=='True' for r in non)
bars(a[0],['En contacto 26','Sin contacto 26'],[contact,len(non)-contact],['#259f94','#dd734a']);a[0].set_ylabel('Fragmentos no principales');a[0].set_title(f'Contacto: {100*contact/len(non):.1f}%')
for k,col,label in zip(regions,colors,names):
    d=[float(r['dist_mm_gt']) for r in non if r['region']==k and r['en_contacto']=='False'];a[1].hist(d,bins=np.arange(0,81,5),alpha=.55,color=col,label=f'{label} (n={len(d)})')
a[1].set_xlabel('Distancia EDT (mm)');a[1].set_ylabel('Fragmentos');a[1].set_title('Solo fragmentos sin contacto 26');a[1].legend(fontsize=10)
f.text(.08,.105,'EDT registrada: distancia entre centros de vóxeles; el contacto no implica necesariamente distancia cero.',fontsize=10)
save(f,'eda_07_contacto_separacion','eda_fragmentos.csv')

f,a=fig('Geometría física de las imágenes','El espaciado del vóxel es necesario para interpretar distancias y volúmenes.')
spacing=[[float(c[k]) for c in C] for k in ['sx_mm','sy_mm','sz_mm']]
a[0].boxplot(spacing,tick_labels=['X','Y','Z']);a[0].set_ylabel('Espaciado (mm)');a[0].set_title('Espaciado por eje')
a[1].scatter([int(c['nx']) for c in C],[int(c['ny']) for c in C],color='#3e83be',alpha=.6,s=55);a[1].set_xlabel('Ancho original (píxeles)');a[1].set_ylabel('Alto original (píxeles)');a[1].set_title('Dimensiones de los cortes originales');save(f,'eda_08_geometria','eda_casos.csv')

if H:
    f,a=fig('Intensidades HU dentro del hueso anotado',f'Análisis disponible para {len(H)} casos; no se extrapola a pacientes no incluidos en este archivo.')
    a[0].hist([float(r['hueso_p50']) for r in H],bins=12,color='#3e83be');a[0].set_xlabel('Mediana HU por caso');a[0].set_ylabel('Casos');a[0].set_title('Intensidad central por paciente')
    a[1].hist([100*float(r['frac_hueso_bajo_200HU']) for r in H],bins=12,color='#dd734a');a[1].set_xlabel('Vóxeles óseos anotados bajo 200 HU (%)');a[1].set_ylabel('Casos');a[1].set_title('Qué puede excluir un umbral de 200 HU');save(f,'eda_09_intensidades_hu','eda_hu_hueso.csv')
print('Gráficas PNG y SVG guardadas en',OUT)
