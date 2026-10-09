"""Mini-paper, model cards y evidencia visual derivados de resultados guardados."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pengwin.instances import semantic_labels

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def pct(x):return 'No evaluable' if x is None else f'{x:.2%}'
def main():
    out=ROOT/'salidas/cierre';doc=ROOT/'doc';doc.mkdir(exist_ok=True)
    model=read(out/'seleccion_modelo.json');cal=read(out/'ajuste/comparacion.json');boxes=read(out/'ajuste_cajas/comparacion.json');inst=read(out/'calibracion_instancias.json');ab=read(out/'ablaciones/comparacion.json');test=read(out/'comparacion_test/metricas.json');vol=read(out/'evaluacion_volumetrica_test.json');lat=read(out/'latencia.json');frozen=read(out/'protocolo_final.json')
    # Matrices de clasificación multietiqueta: presencia/ausencia por región.
    names=['Sacro','Coxal izquierdo','Coxal derecho'];fig,axs=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
    for ax,r,name in zip(axs,test['classification_own']['per_class'],names):
        cm=np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]])
        ax.imshow(cm,cmap='Blues');ax.set_title(name);ax.set_xticks([0,1],['Ausente','Presente']);ax.set_yticks([0,1],['Ausente','Presente']);ax.set_xlabel('Predicción');ax.set_ylabel('Referencia')
        for (i,j),value in np.ndenumerate(cm):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>cm.max()/2 else 'black')
    fig.savefig(out/'matriz_clasificacion.png',dpi=150);plt.close(fig)
    # Cinco peores cortes centrales de la comparación SAM, criterio fijado: Dice anatómico.
    examples=[]
    for p in sorted((out/'comparacion_test').glob('*.npz')):
        with np.load(p) as a:data={k:a[k].copy() for k in a.files}
        gt=semantic_labels(data['gt_instances']);pred=data['own_semantic'];scores=[]
        for r in (1,2,3):
            den=np.count_nonzero(gt==r)+np.count_nonzero(pred==r)
            if den:scores.append(2*np.count_nonzero((gt==r)&(pred==r))/den)
        score=float(np.mean(scores)) if scores else 1.
        fp=np.count_nonzero((pred>0)&(gt==0));fn=np.count_nonzero((pred==0)&(gt>0));wrong=np.count_nonzero((pred>0)&(gt>0)&(pred!=gt))
        hypothesis='Predomina tejido sobrante; revisar especificidad y límites de las cajas.' if fp>fn else 'Predominan omisiones; revisar cobertura de cortes y cajas que recortan hueso.'
        if wrong>max(1,np.count_nonzero(gt>0))*.1:hypothesis+=' Hay confusión entre regiones anatómicas.'
        examples.append((score,p.stem,data,gt,{'case_slice':p.stem,'dice_semantic':score,'false_positive_pixels':int(fp),'false_negative_pixels':int(fn),'wrong_region_pixels':int(wrong),'hypothesis':hypothesis}))
    examples.sort(key=lambda x:x[0]);worst=examples[:5];cmap=ListedColormap(['black','#ef714a','#58a9ff','#36c7aa'])
    fig,axs=plt.subplots(5,3,figsize=(12,18),layout='constrained')
    for axes,(score,name,data,gt,details) in zip(axs,worst):
        for ax,(title,mask) in zip(axes,[('Referencia',gt),('Modelo propio',data['own_semantic']),('SAM',data['sam_semantic'])]):
            ax.imshow(data['image'],cmap='gray',vmin=0,vmax=1);ax.imshow(np.ma.masked_equal(mask,0),cmap=cmap,vmin=0,vmax=3,alpha=.55);ax.set_title(f'{name} · {title}\nDice propio: {score:.1%}');ax.axis('off')
    fig.savefig(out/'cinco_peores.png',dpi=140);plt.close(fig)
    (out/'cinco_peores.json').write_text(json.dumps([x[4] for x in worst],indent=2,ensure_ascii=False),encoding='utf-8')
    calrows='\n'.join(f"| {r['name']} | {pct(r['metrics']['semantic']['macro_dice'])} | {pct(float(np.mean(r['boundary_f1'])))} | {r['epoch']} |" for r in cal['variants'])
    abrows='\n'.join(f"| {r['name']} | {pct(r['metrics']['semantic']['macro_dice'])} | {pct(r['metrics']['detection']['mAP50'])} | {pct(r['metrics']['classification']['macro_f1'])} |" for r in ab['results'])
    volrows='\n'.join(f"| {r['case']} | {r['gt']} | {r['pred']} | {r['missed']} | {r['extra']} | {pct(r['dice_gt'])} | {pct(r['iou_gt'])} | {pct(r['dice_symmetric'])} | {r['distance_valid_pairs']} | {r['distance_MAE_mm'] if r['distance_MAE_mm'] is not None else 'No evaluable'} |" for r in vol['cases'])
    valvol=read(out/'evaluacion_volumetrica_val.json')
    valrows='\n'.join(f"| {r['case']} | {r['gt']} | {r['pred']} | {r['missed']} | {r['extra']} | {pct(r['dice_gt'])} | {pct(r['dice_symmetric'])} |" for r in valvol['cases'])
    val=model['metrics'];det=test['detection_own'];cl=test['classification_own'];sm=test['semantic']
    pairs=sum(r['distance_valid_pairs'] for r in vol['cases']);errors=[]
    for r in vol['cases']:errors.extend(x['absolute_error_mm'] for x in read(out/f"volumenes/{r['case']}/comparacion_distancias.json") if x['valid_pair'])
    mae=float(np.mean(errors)) if errors else None
    text=f'''# PENGWIN: integración y evaluación del pipeline local

## Resumen

Se integró un sistema propio de detección, segmentación anatómica, reconstrucción de instancias y medición física. El desarrollo conserva FundidoraPC, CBAM y tres cabezas principales, con auxiliares de bordes e interiores. Los experimentos adicionales separan calibración del decoder, ajuste de cajas, posprocesado volumétrico y ablaciones de arquitectura. Los resultados se presentan sin sustituir métricas de fragmentos por métricas de regiones. El desempeño observado no equivale a validación clínica.

Equipo de tres integrantes autorizado por el profesor, según confirmación del usuario. Esta excepción no se contabiliza como incumplimiento.

## Datos y protocolo

100 CT con anotaciones originales PENGWIN en MetaImage; 70 pacientes train, 15 val y 15 test con partición fija. Entrenamiento: ocho cortes uniformes por paciente, 560 train y 120 val, resolución 256², lote 8, semilla 42. No se entrenó sobre todos los cortes ni se generaron etiquetas con SAM. La geometría se reorienta a LPS preservando coordenadas físicas y espaciado; ventana HU [−500,1300]. Los tests sintéticos comprueban funciones geométricas y no forman parte del entrenamiento.

La selección de red y posprocesado utiliza validación. El protocolo se congeló en `salidas/cierre/protocolo_final.json` antes de las nuevas evaluaciones test: detección/clasificación en 45 cortes (10/50/90 % de cada uno de los 15 pacientes); SAM y segmentación 2D en los 15 cortes centrales; reconstrucción completa en los tres primeros pacientes test, definidos antes de ejecutar. **El test tuvo exposición exploratoria en avances anteriores:** no debe presentarse como un conjunto nunca visto por el equipo. Las cifras son una evaluación adicional con ese límite, sin reajustar tras esta ejecución.

## Arquitectura y decisiones

FundidoraPC compartida con atención CBAM de canal y espacio antes de las tres cabezas: clasificación anatómica multietiqueta, detector por grid con NMS propio y segmentación. El decoder mantiene ocho canales, skips aditivas y supervisión profunda. La salida semántica tiene fondo y tres huesos; los IDs locales de fragmento no se convierten en clases globales. Las instancias se reconstruyen dentro de cada región. No se utilizan YOLO, Detectron2 ni Mask R-CNN como arquitectura final; ResNet18 aparece solo en el estudio de transferencia de backbone, con cabezas propias aleatorias.

Se congelaron backbone, clasificador y detector al calibrar el decoder; se comprobaron sus tensores inmutables tras cada época. El ajuste posterior de cajas solo actualiza la cabeza detectora y añade GIoU propio. No se acepta si empeora el mAP50 o el Dice de alguna región respecto al modelo previo; su efecto combina entrenamiento adicional y GIoU, no los aísla.

## Pérdidas y calibración

La fase multitarea original pondera objetidad 2, regresión de caja 5, clase de caja 1, presencia 1 y segmentación 1,2. La mayor ponderación de caja compensa la escala de regresión normalizada; la focal equilibra celdas positivas y negativas. Segmentación combina CE, Dice 1,5, interfaces 0,5, interiores 0,5 y supervisión profunda 0,3. La motivación es el desbalance de fondo y la necesidad de mantener anatomía e interfaces separadas. Estos coeficientes originales se conservaron; **no se realizó una búsqueda exhaustiva de todos los pesos multitarea**.

La calibración de esta revisión usa cuatro épocas adicionales iguales por variante, LR 0,0002, optimizador AdamW y mismos lotes. Se prueban (peso CE de fondo, peso macro, peso contorno): control (0,2;0,5;0,2), precisión (1;0,25;0,5), contorno (0,5;0,25;1). Se aumenta la penalización del fondo para responder al exceso de tejido predicho, y se contrasta el peso del contorno debido al F1 de borde insuficiente. Selección: 0,7 Dice + 0,3 F1 de contorno a dos píxeles, sin empeorar ninguna región frente al inicio. Es una calibración sobre val, no un resultado test.

| Variante decoder | Dice semántico val | F1 medio de contorno | Época elegida |
|---|---:|---:|---:|
{calrows}

La selección de decoder fue **{cal['selected']['name']}**. Ajuste de cajas aceptado: **{bool(boxes['accepted'])}**. Se evita atribuir toda mejora a un operador aislado. El refinamiento bilateral de la revisión anterior se deja apagado en esta fase: su efecto fue marginal y se prioriza corregir la predicción aprendida.

Comparación visual en tres posiciones fijas (25/50/75 %) del caso 002 de validación. La imagen anterior incluye su refinamiento seleccionado; las tablas de calibración usan la comparación controlada sin ese refinamiento. Se observan contornos más ajustados en algunas zonas, pero también recortes por cajas y confusiones anatómicas que persisten.

![Comparación visual de validación](../salidas/cierre/comparacion_visual_val.png)

## Ablaciones requeridas

Tres épocas por ejecución, misma partición, inicialización de cabezas comprobada por hash dentro de cada par, LR 0,0003 y lote 8. Fundidora con/sin CBAM evalúa atención. ResNet18 con/sin pesos ImageNet evalúa transferencia dentro de la misma arquitectura. Las cabezas se entrenan desde cero en ambos ResNet. No se interpreta la diferencia Fundidora/ResNet como efecto de transferencia.

| Variante | Dice semántico val | mAP50 val | F1 clasificación val |
|---|---:|---:|---:|
{abrows}

Es un estudio corto y controlado, no una demostración de convergencia. CBAM se mantiene en la arquitectura final porque es un requisito; los resultados no se fuerzan a favorecerlo.

## Resultados de prueba y SAM

| Métrica | Resultado | Objetivo sugerido |
|---|---:|---:|
| F1 clasificación por corte | {pct(cl['macro_f1'])} | 85 % |
| AUC clasificación por corte | {pct(cl['macro_auc'])} | 85 % |
| IoU de cajas incluyendo omisiones | {pct(det['mean_gt_iou_at_conf025'])} | 65 % |
| mAP50 | {pct(det['mAP50'])} | 65 % |
| mAP50:95 | {pct(det['mAP50_95'])} | 40 % |

Clasificación mide presencia anatómica por corte, no exactitud de cada fragmento. La definición de IoU de cajas incluye GT omitidos como cero. Los objetivos del enunciado son sugeridos; no se afirman alcanzados cuando no lo están.

SAM ViT-B recibe exclusivamente cajas predichas tras NMS; no recibe cajas ni máscaras GT, no se ajusta y no participa en el pipeline final.

| 15 cortes centrales test | Dice anatómico | IoU anatómico |
|---|---:|---:|
| Modelo propio | {pct(sm['propio']['macro_dice'])} | {pct(sm['propio']['macro_iou'])} |
| SAM zero-shot | {pct(sm['sam']['macro_dice'])} | {pct(sm['sam']['macro_iou'])} |

Estas cifras anatómicas no son el Dice por fragmento exigido. La comparación también guarda instancias 2D con asignación 1:1, omisiones y extras en `comparacion_test/metricas.json`; sus filtros se expresan en píxeles 2D y no se confunden con mm³. SAM no está diseñado aquí para separar todos los fragmentos que comparten una caja anatómica.

![Matrices de clasificación por presencia](../salidas/cierre/matriz_clasificacion.png)

## Fragmentos 3D y distancias

Se compararon siete políticas sobre tres volúmenes val: interiores aprendidos, componentes conectados, interfaces aprendidas con tres umbrales y dos filtros de volumen adicionales. Se selecciona por Dice simétrico que penaliza extras y omisiones; no se fuerza un máximo de diez predicciones ni se borra GT. Política congelada: `{frozen['instance_policy']}`. Un umbral mínimo de volumen puede descartar fragmentos verdaderos pequeños; esa consecuencia cuenta en las métricas.

Resultados sobre validación, utilizada para seleccionar la política:

| Caso val | GT | Predichos | Omitidos | Extras | Dice GT | Dice simétrico |
|---|---:|---:|---:|---:|---:|---:|
{valrows}

En el caso 002, la versión anterior produjo 96 instancias para seis fragmentos GT, con dos omitidos y 92 extras; Dice GT 32,30 % y Dice simétrico 1,98 %. La nueva versión reduce extras pero puede aumentar omisiones. No es una mejora uniforme y este caso interviene en la selección.

| Caso test | GT | Predichos | Omitidos | Extras | Dice GT | IoU GT | Dice simétrico | Pares distancia | MAE mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{volrows}

Dice GT promedia los fragmentos de referencia, incluyendo omitidos como cero; no penaliza directamente cada predicción extra. Dice simétrico incluye además esas predicciones sin correspondencia como cero y revela la sobresegmentación. El objetivo sugerido de Dice por fragmento 85 % / IoU 70 % no debe confundirse con el Dice anatómico.

Pares de distancia válidos totales: **{pairs}**. MAE agregado: **{str(round(mae,4))+' mm' if mae is not None else 'no evaluable'}**. Solo se comparan distancias si fragmento y principal se emparejan correctamente con IoU ≥ 0,5. Un valor no evaluable nunca se transforma en cero. El principal predicho es la instancia mayor de la región; el GT usa IDs 1/11/21. La distancia se calcula mediante `distance_transform_edt` con espaciado físico, entre centros de vóxeles superficiales; es una aproximación discreta de borde a borde. Contacto 26 se informa aparte. La reducción XY a 256 conserva campo físico y todos los cortes Z, pero puede perder fragmentos pequeños.

## Cinco peores casos y análisis de errores

Se seleccionan automáticamente los cinco peores **cortes centrales** entre los 15 evaluados con SAM, por Dice anatómico. No representan necesariamente los cinco peores volúmenes completos. Las hipótesis se basan en falsos positivos, omisiones y confusión entre regiones, sin atribuir diagnósticos clínicos.

{chr(10).join('- '+x[4]['case_slice']+': '+x[4]['hypothesis'] for x in worst)}

![Cinco peores cortes](../salidas/cierre/cinco_peores.png)

## Visualización, reproducción y cómputo

El dashboard integra MIP del CT por umbral HU sin modelo, recorrido de todos los cortes con cajas/máscaras y distancias sobre la imagen, y malla 3D con marching cubes y Plotly Mesh3d, coloreada por hueso y etiquetada por fragmento. La malla se simplifica solo para visualizar; las distancias usan el volumen. Los casos mostrados son resultados ya calculados, no inferencia interactiva sobre archivos nuevos. Es accesible desde un navegador y tiene disposición adaptable a pantalla estrecha.

CPU: {lat['results']['cpu']['mean_ms']:.2f} ms/corte de media; p95 {lat['results']['cpu']['p95_ms']:.2f} ms, lote 1, cinco warmups y {lat['results']['cpu']['n']} cortes reales. Incluye forward, NMS, restricción de máscaras y transferencia a CPU; excluye IO, resize y reconstrucción 3D. GPU: **{lat['results']['cuda']['status']}**. El entorno instalado es CPU y no se encontró `nvidia-smi`; no se declara que el equipo carezca físicamente de GPU. Falta medir en un entorno CUDA funcional. La prueba está en `scripts/medir_latencia.py`.

Las rutas y comandos están en README, los pesos conservan hashes y los paquetes exactos están en `requirements-reproducible.txt`. GitHub, protección de main y revisiones por integrante no se verificaron ni modificaron por instrucción del usuario. El túnel de Cloudflare es temporal y solo queda cumplido cuando exista una URL comprobada; consultar el registro de despliegue.

## Limitaciones y uso responsable

Uso exclusivamente académico. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales. Persisten errores de segmentación, fusiones de fragmentos en contacto, ruido y sensibilidad a cortes poco representados. No se garantiza alcanzar Dice de fragmento ≥ 85 % ni IoU ≥ 70 %. El muestreo pequeño, el test previamente explorado y las ablaciones de tres épocas limitan generalización. Las distancias deben interpretarse junto con cobertura de emparejamiento y calidad de segmentación, nunca de forma aislada.

## Referencias y procedencia

- Enunciado local `Proyecto_Corte2_PENGWIN.docx`, secciones 3–7.
- Notebook de Sesión 2 del profesor, adaptado en la revisión anterior; no ejecutado como pipeline final de YOLO/Mask R-CNN.
- Pesos oficiales de backbone ImageNet: https://download.pytorch.org/models/resnet18-f37072fd.pth (SHA256 registrado en ablaciones).
- SimpleITK DICOMOrient: https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1DICOMOrientImageFilter.html
- Cloudflare Quick Tunnels: https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/
'''
    (doc/'INFORME_CIERRE.md').write_text(text,encoding='utf-8')
    cards=doc/'model_cards';cards.mkdir(exist_ok=True)
    for config_file in sorted((ROOT/'salidas').rglob('config.json')):
        cfg=read(config_file);folder=config_file.parent;weights=[p for p in (folder/'best.pth',folder/'final.pth') if p.exists()]
        if not weights:continue
        name=folder.relative_to(ROOT/'salidas').as_posix().replace('/','_')
        metrics=torch.load(weights[0],map_location='cpu',weights_only=False).get('val')
        card=f'''# Model card: {name}

Uso exclusivamente académico. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Arquitectura y entrenamiento

Tres cabezas propias de clasificación, detección y segmentación. Consultar la configuración exacta; las variantes ResNet usan backbone natural opcional y cabezas nuevas. Decoder de ocho canales. Datos: PENGWIN CT, 70 pacientes train, 15 val, 15 test; 560 cortes train y 120 val, 256², semilla 42. No etiquetas sintéticas ni SAM como entrenador. Los resultados de test del informe corresponden únicamente al modelo congelado seleccionado, no a todas las variantes.

Configuración guardada:
```json
{json.dumps(cfg,indent=2,ensure_ascii=False)}
```

## Pesos

{chr(10).join('- '+p.relative_to(ROOT).as_posix()+' — SHA256 '+hashlib.sha256(p.read_bytes()).hexdigest() for p in weights)}

## Métricas de esta variante

```json
{json.dumps(metrics,indent=2,ensure_ascii=False) if metrics is not None else 'Consultar historial.json/comparacion.json de la misma carpeta; no atribuir métricas de otro checkpoint.'}
```

## Límites

Puede fusionar fragmentos que se tocan en la superficie de fractura o inventar componentes. El número de cortes entrenados es reducido; la validación se usa para seleccionar y no es una evaluación independiente. El test tuvo exposición exploratoria previa. Distancias discretas con espaciado físico; errores de máscara se transmiten a las medidas. Sin validación clínica. GPU no medida en este equipo. Reproducir con los scripts y el entorno registrados en el informe.
'''
        (cards/f'{name}.md').write_text(card,encoding='utf-8')
    final_card=f'''# Model card del modelo final seleccionado

Uso exclusivamente académico. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

Arquitectura: FundidoraPC + CBAM, tres cabezas propias, decoder con skips y ocho canales. Backbone inicialmente desde cero, sin pesos de SAM ni de detectores externos. Entrenamiento y selección en 70/15 pacientes; resultados de prueba limitados a los cortes y volúmenes descritos en el informe.

Checkpoint: `{frozen['checkpoint']}`. SHA256: `{frozen['checkpoint_sha256']}`.

Clasificación test: F1 {pct(cl['macro_f1'])}, AUC {pct(cl['macro_auc'])}. Detección test: mAP50 {pct(det['mAP50'])}, mAP50:95 {pct(det['mAP50_95'])}. Dice anatómico en 15 cortes centrales: {pct(sm['propio']['macro_dice'])}; no equivale a Dice por fragmento. Resultados volumétricos y distancias en `INFORME_CIERRE.md`.

Política de instancias: `{frozen['instance_policy']}`. Puede omitir fragmentos pequeños y fusionar fragmentos en contacto; no forzar resultados a la taxonomía esperada. Distancia EDT entre centros de vóxeles superficiales, con spacing, y comparación válida solo con correspondencia de fragmento y principal.

No usar para decisiones clínicas. Test previamente explorado, pocos cortes de entrenamiento, rendimiento insuficiente en casos difíciles, GPU pendiente de medir. Las alertas son control de calidad y no diagnóstico. La reproducción usa la configuración y los hashes guardados; no modificar el protocolo tras observar test.
'''
    (doc/'MODEL_CARD_FINAL.md').write_text(final_card,encoding='utf-8')
    print('Informe, model cards y galería generados',flush=True)
if __name__=='__main__':main()
