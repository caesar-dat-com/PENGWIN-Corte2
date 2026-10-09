"""Genera un informe a partir de registros existentes, sin inventar ejecuciones."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from pengwin.models.detector import PelvisDetector
from pengwin.instances import semantic_labels
from pengwin.metrics_avance3 import gate_semantic
from entrenar_pacientes import decode

OUT=ROOT/'salidas/revision_oct08'
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def percent(x):return 'No disponible' if x is None else f'{100*x:.2f} %'

def main():
    torch.set_num_threads(2)
    hist=load(OUT/'entrenamiento/historial.json');m=load(OUT/'entrenamiento/metricas_val.json')
    ck=torch.load(OUT/'entrenamiento/best.pth',map_location='cpu',weights_only=False)
    model=PelvisDetector(pretrained=False);model.load_state_dict(ck['model']);model.eval()
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained');epochs=[r['epoch'] for r in hist]
    axs[0].plot(epochs,[r['loss']['total'] for r in hist],marker='o');axs[0].set(title='Pérdida de entrenamiento',xlabel='Época',ylabel='Pérdida total')
    axs[1].plot(epochs,[r['val']['detection']['mAP50'] for r in hist],marker='o',label='mAP50 cajas')
    axs[1].plot(epochs,[r['val']['semantic']['macro_dice'] for r in hist],marker='o',label='Dice semántico');axs[1].set(title='Validación: pacientes distintos',xlabel='Época',ylim=(0,1));axs[1].legend()
    fig.savefig(OUT/'entrenamiento.png',dpi=160);plt.close(fig)
    # Primeros tres pacientes val en orden fijado, corte central de la muestra uniforme.
    cases=load(ROOT/'splits/splits.json')['val'][:3];fig,axs=plt.subplots(3,3,figsize=(12,11),layout='constrained')
    colors=['#fa6046','#2586ef','#14b899'];cmap=ListedColormap(['black']+colors)
    with torch.inference_mode():
        for row,cid in enumerate(cases):
            with np.load(OUT/f'datos/{cid}.npz') as data:
                j=len(data['z'])//2;x=data['images'][j].copy();ids=data['instances'][j].copy();gtboxes=data['boxes'][j].copy();z=int(data['z'][j])
            pred=model(torch.from_numpy(x)[None,None].repeat(1,3,1,1));b,s,c=[v.numpy() for v in decode(pred)]
            sem=gate_semantic(pred['mascaras'][0].softmax(0).numpy(),b,c)
            for ax in axs[row]:ax.imshow(x,cmap='gray',vmin=0,vmax=1);ax.axis('off')
            axs[row,0].imshow(np.ma.masked_equal(semantic_labels(ids),0),cmap=cmap,vmin=0,vmax=3,alpha=.55)
            axs[row,1].imshow(np.ma.masked_equal(sem,0),cmap=cmap,vmin=0,vmax=3,alpha=.55)
            for bb in gtboxes[gtboxes[:,0]>=0]:
                k=int(bb[0]);x1,y1,x2,y2=bb[1:]*256;axs[row,2].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=colors[k],linewidth=1.2))
            for box,score,k in zip(b,s,c):
                x1,y1,x2,y2=box*256;axs[row,2].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=colors[int(k)],linestyle='--',linewidth=1.2))
            axs[row,0].set_title(f'Caso {cid} · z={z} · GT');axs[row,1].set_title('Segmentación propia');axs[row,2].set_title('Cajas: GT continua / pred. discontinua')
    fig.savefig(OUT/'validacion_ejemplos.png',dpi=160);plt.close(fig)
    lines=['# Revisión local del proyecto PENGWIN — 8 de octubre de 2026','',
      'Se revisó y amplió la nueva versión local. Se mantuvieron el backbone FundidoraPC, CBAM espacial de 9 × 9 y el decodificador de ocho canales internos. No se consultó ni actualizó Git.','',
      '## Qué cambió y por qué','',
      '| Antes | Cambio verificado |','|---|---|',
      '| Porcentajes supuestos de ruido | Conteo directo de las 100 máscaras: 255 de 62.738 cajas se excluirían con área < 15 (0,406 %), no 1.506. |',
      '| Área pequeña se describía como ruido clínicamente seguro | Umbral por defecto 0. Se conservan las anotaciones; el filtro no identifica componentes ni artefactos. |',
      '| Cuatro cortes del mismo caso para aprender y evaluar | 560 cortes de 70 pacientes train y 120 de 15 pacientes val. Test separado. |',
      '| Máscara de tres huesos confundida con fragmentos | IDs GT conservados y cabeza auxiliar de interfaces; reconstrucción de instancias por watershed 3D. |',
      '| Distancia 2D sacro–coxal | Distancia 3D fragmento–principal del mismo hueso, con espaciado físico y emparejamiento contra GT. |',
      '| Rutas data/raw inexistentes en este equipo | Configuración local de los volúmenes extraídos, búsqueda recursiva y control de geometría. |',
      '| NMS creaba tensores en CPU aunque la entrada fuera GPU | Se conserva el dispositivo. Prueba GPU pendiente: equipo disponible solo CPU. |',
      '| Dos objetivos podían sobrescribirse en una celda | Se detecta la colisión y se informa; no se elimina silenciosamente GT. |','',
      '## Entrenamiento ejecutado','',
      f'- {len(hist)} épocas, CPU, semilla 42, lote 8. Mejor checkpoint por validación: época {ck["epoch"]}.',
      '- Ocho cortes uniformes por paciente, incluidos cortes sin hueso. Son tomografías reales: no se entrenó con imágenes sintéticas.',
      '- Selección: media de mAP50 y Dice semántico de validación. Las máscaras quedan limitadas a cajas predichas tras NMS propio.',
      '- Se preservó el split congelado 70/15/15. No es entrenamiento sobre todos los cortes de los 70 volúmenes.',
      f'- Pérdida media: {hist[0]["loss"]["total"]:.4f} → {hist[-1]["loss"]["total"]:.4f}.','',
      '| Métrica en validación | Resultado |','|---|---|',
      f'| mAP50 (cajas) | {percent(m["detection"]["mAP50"])} |',
      f'| mAP50:95 (cajas) | {percent(m["detection"]["mAP50_95"])} |',
      f'| IoU media por caja GT, omisiones = 0 | {percent(m["detection"]["mean_gt_iou_at_conf025"])} |',
      f'| F1 macro de presencia de regiones | {percent(m["classification"]["macro_f1"])} |',
      f'| AUC macro de presencia | {percent(m["classification"]["macro_auc"])} |',
      f'| Dice semántico macro | {percent(m["semantic"]["macro_dice"])} |','',
      f'Dice por región: sacro {percent(m["semantic"]["dice"][0])}; coxal izquierdo {percent(m["semantic"]["dice"][1])}; coxal derecho {percent(m["semantic"]["dice"][2])}. Un Dice cero indica una clase no segmentada correctamente; no declarar el modelo completo.', '',
      'Las clases de presencia son sacro/coxal izquierdo/coxal derecho, no diagnóstico de fractura. AP usa interpolación de 101 puntos y no todos los filtros del evaluador COCO. Dice semántico no mide separación de fragmentos.',
      '', '![Curvas](../salidas/revision_oct08/entrenamiento.png)','![Validación](../salidas/revision_oct08/validacion_ejemplos.png)','',
      '## Pérdida e instancias','',
      'Se conserva la pérdida de detección con pesos 2/5/1/1 (objetidad/caja/clase/presencia) y Focal gamma 0,5. En el experimento nuevo se promedian por separado positivos y negativos para que las celdas de fondo no dominen. Se añade 1,2 × (CE ponderada + 1,5 × Dice + 0,5 × BCE de interfaces). La CE asigna peso 0,2 al fondo y 1 a cada hueso; interfaces e interiores anotados se equilibran por separado.',
      '', 'Estos son pesos iniciales razonados, no una calibración comparativa terminada. Falta comparar configuraciones sobre train/val antes de declarar ese requisito cumplido. Los identificadores de fragmentos pueden cambiar entre pacientes; la pérdida de interfaces no depende de su número concreto.',
      '', 'La reconstrucción usa conectividad 26, umbral de borde 0,5 y volumen mínimo predicho de 20 mm³. Este mínimo afecta solamente las predicciones, nunca las anotaciones GT; no tiene validación clínica. Los bordes se aprenden en 2D y luego se apilan: no es una red volumétrica. Un borde incompleto puede unir fragmentos y un borde excesivo puede dividirlos.',
      '', '## Prueba volumétrica y distancia','']
    volumes=sorted((OUT/'volumenes').glob('*/resumen.json')) if (OUT/'volumenes').exists() else []
    if not volumes:lines+=['Implementada, aún sin resultado volumétrico guardado.']
    for path in volumes:
        v=load(path);f=v['fragment_metrics']
        lines += [f'Caso {v["case"]} ({v["split"]}): {v["slices_contiguous"]} cortes contiguos. {f["gt_fragments"]} fragmentos GT y {f["pred_instances"]} instancias predichas. Dice macro por fragmento GT, incluidas omisiones: {percent(f["dice_gt_macro_with_misses"])}. Omisiones: {f["missed_gt"]}; predicciones adicionales: {f["extra_pred"]}.', '',f'Parejas válidas para distancia: {f["distance_comparable_pairs"]}; MAE: {f["distance_MAE_mm"] if f["distance_MAE_mm"] is not None else "no calculable, sin parejas válidas"}. Latencia media del modelo, NMS y ensamblado semántico por corte: {v["latency_mean_ms"]:.1f} ms en {v["device"]}. Excluye lectura, redimensionado y watershed 3D.','']
    lines += ['GT principal: etiquetas 1/11/21. Principal predicho: instancia más voluminosa del mismo hueso. Se exige IoU ≥ 0,5 tanto para fragmento como para principal antes de comparar distancias. Las distancias se aproximan entre centros de vóxeles de superficie mediante EDT; no son distancias exactas entre caras. El contacto por vecindad 26 se informa aparte. Se ajustan espaciado y origen al redimensionar XY a 256; se mantienen todos los cortes Z. Las máscaras GT se remuestrean por vecino más cercano a esta cuadrícula; fragmentos muy pequeños pueden desaparecer, por lo que estas métricas no sustituyen una evaluación a resolución nativa.','', '## Comparación SAM','']
    if (OUT/'comparacion_test/metricas.json').exists():
        s=load(OUT/'comparacion_test/metricas.json')
        snapshots=sorted((OUT/'comparacion_test').glob('*.npz'))[:3]
        if snapshots:
            fig,axs=plt.subplots(len(snapshots),3,figsize=(12,3.6*len(snapshots)),squeeze=False,layout='constrained')
            for row,path in enumerate(snapshots):
                with np.load(path) as a:
                    for col,(key,title) in enumerate([('gt_instances','GT'),('own_semantic','Modelo propio'),('sam_semantic','SAM con cajas predichas')]):
                        mask=semantic_labels(a[key]) if key=='gt_instances' else a[key]
                        axs[row,col].imshow(a['image'],cmap='gray',vmin=0,vmax=1)
                        axs[row,col].imshow(np.ma.masked_equal(mask,0),cmap=cmap,vmin=0,vmax=3,alpha=.55)
                        axs[row,col].set_title(f'{path.stem} · {title}');axs[row,col].axis('off')
            fig.savefig(OUT/'comparacion_sam.png',dpi=160);plt.close(fig)
            lines += ['![Comparación SAM](../salidas/revision_oct08/comparacion_sam.png)','']
        lines += [f'Evaluación congelada: {s["slices"]} cortes (3 por paciente test); comparación de segmentación en {s["sam_comparison_slices"]} cortes centrales (1 por paciente). No es evaluación de todos los volúmenes test.', '', '| Métrica en los mismos 15 cortes | Propio | SAM ViT-B zero-shot |','|---|---|---|',f'| Dice semántico macro | {percent(s["semantic"]["propio"]["macro_dice"])} | {percent(s["semantic"]["sam"]["macro_dice"])} |',f'| IoU semántico macro | {percent(s["semantic"]["propio"]["macro_iou"])} | {percent(s["semantic"]["sam"]["macro_iou"])} |',f'| Dice por fragmento 2D GT, incluidas omisiones | {percent(s["instances_2d"]["propio"]["dice_gt_macro_with_misses"])} | {percent(s["instances_2d"]["sam"]["dice_gt_macro_with_misses"])} |','', f'Detección propia en 45 cortes: mAP50 {percent(s["detection_own"]["mAP50"])}, mAP50:95 {percent(s["detection_own"]["mAP50_95"])}. Clasificación: F1 {percent(s["classification_own"]["macro_f1"])}, AUC {percent(s["classification_own"]["macro_auc"])}.', '', f'Instancias 2D adicionales: propio {s["instances_2d"]["propio"]["extra_pred"]}; SAM {s["instances_2d"]["sam"]["extra_pred"]}. Dice simétrico que también penaliza esas predicciones: propio {percent(s["instances_2d"]["propio"]["dice_symmetric_macro_with_unmatched"])}; SAM {percent(s["instances_2d"]["sam"]["dice_symmetric_macro_with_unmatched"])}. El Dice calculado solo sobre GT puede ocultar la sobresegmentación.', '', 'SAM recibió únicamente cajas del modelo, nunca cajas GT. Se conservaron omisiones y falsos positivos. La comparación 2D usa máscaras agrupadas por anatomía y componentes conectados para SAM: no constituye un segmentador de fragmentos 3D. El test ya tuvo exposición exploratoria en avances previos; no es un conjunto externo completamente virgen. No se ajustó el modelo con estos resultados.']
    else:lines+=['Script preparado; ejecución completa aún no registrada.']
    lines += ['', '## Qué falta para cerrar el avance 3','',
      '1. Mejorar y ampliar el entrenamiento; estudiar errores por hueso y fragmentos pequeños.',
      '2. Calibrar pesos y posprocesamiento exclusivamente con train/val, con comparación explícita.',
      '3. Evaluar más volúmenes completos y aumentar la cobertura de pares fragmento–principal con correspondencias fiables.',
      '4. Presentar métricas de instancia y distancia junto con omisiones; no sustituirlas por Dice semántico.',
      '5. Medir rendimiento GPU cuando exista equipo disponible y abordar visualizadores/dashboard del siguiente avance.',
      '', '## Evidencia y reproducción','',
      '- `salidas/revision_oct08/auditoria.json` y CSV: conteos completos.',
      '- `salidas/revision_oct08/entrenamiento/`: configuración, historial, checkpoints y predicciones de validación.',
      '- `salidas/revision_oct08/volumenes/`: máscaras 3D, emparejamientos y distancias.',
      '- `salidas/revision_oct08/comparacion_test/`: protocolo congelado, hashes y comparación SAM cuando se complete.',
      '- `tests/test_revision.py`: pruebas funcionales y geométricas; registro en `salidas/revision_oct08/pruebas.txt`.',
      '- `GUION_SUSTENTACION_SEMANA9.md`: guion corregido, Yesenia presenta primero.',
      '- `notebooks/historico/Semana2_antes_revision.ipynb`: original preservado, con afirmaciones obsoletas. El cuaderno corregido requiere reejecución.',
      '', 'Consultar README para las órdenes locales. Resultados académicos preliminares; no demuestran validez clínica.']
    (ROOT/'doc').mkdir(exist_ok=True);(ROOT/'doc/REVISION_OCT08.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(ROOT/'doc/REVISION_OCT08.md')
if __name__=='__main__':main()
