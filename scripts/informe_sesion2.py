"""Informe trazable de la adaptación, generado desde métricas medidas."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'salidas/sesion2'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def pct(v):return f'{v*100:.2f} %'

def main():
    data=read(OUT/'comparacion_val.json');m=data['metrics'];old=m['previo'];new=m['sesion2']
    lines=['# Adaptación de Sesión 2 a PENGWIN — semana 10','',
      'Referencia proporcionada por el usuario: `Segmentacion_Sesion2_UNetPP_MaskRCNN_YOLO_SAM_Completo .ipynb`. Se adaptan conceptos del curso; no se ejecutaron sus descargas ni experimentos de mascotas y peatones.','',
      '## Implementación','',
      '- Se conserva FundidoraPC con cuatro etapas residuales y CBAM9, detección grid, NMS propio y clasificación de presencia.',
      '- El decoder recibe características de tres escalas del encoder y de la imagen de entrada. Las proyecciones convierten 128/64/32/3 canales a ocho; después se suman, sin concatenación. Todos los bloques de reconstrucción operan con ocho canales.',
      '- Se añaden tres salidas semánticas auxiliares durante entrenamiento (supervisión profunda). La inferencia usa solo la salida final.',
      '- Los IDs GT se convierten en máscaras individuales, clase anatómica e interiores normalizados por fragmento. Los interiores y las interfaces son objetivos invariantes a renumerar los IDs locales. El interior normalizado es un objetivo 2D en píxeles, no la distancia física entre fragmentos.',
      '- La reconstrucción volumétrica emplea watershed, semillas de interior > 0,5 y borde < 0,5, vecindad 26 y mínimo predicho de 20 mm³. El mínimo no elimina anotaciones GT. Cada instancia conserva la región anatómica.',
      '', 'Esta es una adaptación ligera inspirada en U-Net/U-Net++, no una U-Net++ completa. Los objetivos auxiliares ayudan a representar fragmentos, pero no garantizan separación perfecta. Los parámetros del posprocesamiento aún requieren calibración en validación.',
      '', '## Protocolo ejecutado','',
      'Ambas variantes: entrenamiento desde cero, seis épocas, semilla 42, lote 8 y CPU. Se usan ocho cortes uniformes por paciente: 560 cortes de 70 pacientes train y 120 de 15 val. Se mantiene el split congelado. Se incluyen cortes sin hueso. No se entrenó con imágenes sintéticas.',
      '', 'La comparación es contra el experimento previo de revisión, que usa el decoder sin skips y ya incluye la salida de bordes. No se compara directamente contra el overfit de cuatro cortes del caso 001. Se selecciona cada checkpoint por la media de mAP50 y Dice semántico en val.',
      '', f'Checkpoints seleccionados: previo, época {data["checkpoints"]["previo"]["epoch"]}; adaptación, época {data["checkpoints"]["sesion2"]["epoch"]}. SHA-256 en `salidas/sesion2/comparacion_val.json`.',
      '', '| Métrica de validación | Decoder sin skips | Adaptación Sesión 2 |','|---|---:|---:|']
    for title,group,key in [('mAP50','detection','mAP50'),('mAP50:95','detection','mAP50_95'),('F1 de presencia','classification','macro_f1'),('Dice semántico macro','semantic','macro_dice'),('Dice por instancia GT 2D (omisiones incluidas)','instances_2d','dice_gt_macro'),('Dice simétrico por instancia 2D (incluye extras)','instances_2d','dice_symmetric')]:
        lines.append(f'| {title} | {pct(old[group][key])} | {pct(new[group][key])} |')
    lines+=['', '| Dice por región | Sin skips | Sesión 2 |','|---|---:|---:|']
    for i,name in enumerate(('Sacro','Coxal izquierdo','Coxal derecho')):lines.append(f'| {name} | {pct(old["semantic"]["dice"][i])} | {pct(new["semantic"]["dice"][i])} |')
    lines+=['', f'Instancias GT en los cortes: {new["instances_2d"]["gt"]}. Omisiones previo/nuevo: {old["instances_2d"]["missed"]}/{new["instances_2d"]["missed"]}. Predicciones adicionales previo/nuevo: {old["instances_2d"]["extra"]}/{new["instances_2d"]["extra"]}.',
      '', 'Los IDs se emparejan 1:1 por IoU dentro de la misma región. La métrica simétrica penaliza también instancias adicionales. Son mediciones 2D; no confundirlas con fragmentos volumétricos.',
      '', '![Comparación sobre casos val prefijados](../salidas/sesion2/comparacion_visual.png)',
      '', '## Pérdidas y atribución de la mejora','',
      'Se conserva detección y se añade 1,2 × [CE + 1,5 Dice + 0,5 BCE de interfaces + 0,5 MSE de interiores + 0,3 CE auxiliar]. CE pesa fondo 0,2 y cada hueso 1; las interfaces positivas y los interiores negativos se promedian por separado. La MSE de interiores equilibra hueso/fondo. Las salidas auxiliares se supervisan con etiquetas remuestreadas por vecino más cercano.',
      '', 'El experimento cambia conjuntamente skips, supervisión profunda y objetivo de interiores. No permite atribuir el resultado a un único componente ni sustituye la calibración de pesos o la ablación con/sin CBAM exigida para el informe final.',
      '', '## Inferencia 3D real','']
    path=OUT/'volumenes/002/resumen.json'
    if path.exists():
        v=read(path);f=v['fragment_metrics']
        lines += [f'Caso 002 de validación: {v["slices_contiguous"]} cortes contiguos. {f["gt_fragments"]} fragmentos GT remuestreados y {f["pred_instances"]} instancias predichas. Dice macro por fragmento GT: {pct(f["dice_gt_macro_with_misses"])}; Dice simétrico: {pct(f["dice_symmetric_macro_with_unmatched"])}. Omisiones: {f["missed_gt"]}; adicionales: {f["extra_pred"]}.',
        '', f'Parejas válidas para distancia: {f["distance_comparable_pairs"]}. MAE en mm: {f["distance_MAE_mm"] if f["distance_MAE_mm"] is not None else "no calculable por falta de correspondencias válidas"}.',
        '', 'El principal GT se identifica por 1/11/21; el principal predicho es el de mayor volumen del mismo hueso. Se exige IoU ≥ 0,5 para fragmento y principal. EDT aproxima distancias entre centros de vóxeles de superficie; el contacto26 se informa por separado. XY se reduce a 256 ajustando espaciado y origen. Remuestrear GT puede perder fragmentos pequeños; falta evaluación a resolución nativa.']
    else:lines.append('No hay todavía resultado 3D guardado de esta adaptación.')
    filtered=OUT/'volumenes_filtrado/002/resumen.json'
    if filtered.exists():
        f=read(filtered)['fragment_metrics']
        lines+=['', '### Control del exceso de instancias', '', f'En el mismo caso val 002 se ensayó un mínimo de 50 mm³ para las semillas (se mantiene mínimo de 20 mm³ para instancias predichas). Las instancias bajaron de 1.620 a {f["pred_instances"]}, frente a 6 GT. Dice por fragmento GT: {pct(f["dice_gt_macro_with_misses"])}; simétrico: {pct(f["dice_symmetric_macro_with_unmatched"])}. Siguen existiendo {f["extra_pred"]} predicciones adicionales y {f["distance_comparable_pairs"]} parejas válidas para distancia. Esto continúa siendo sobresegmentación importante; no es una solución validada.', '', 'La inferencia nueva usa este filtro de semillas por defecto. El resultado inicial permanece en volumenes/ y el filtrado en volumenes_filtrado/. Este ajuste se probó solo en un volumen val, no es una evaluación externa.']
    cal=OUT/'calibracion_instancias_2d.json'
    if cal.exists():
        c=read(cal)['selected']
        lines+=['', '### Sensibilidad 2D en validación', '', f'Se compararon nueve combinaciones: semillas mínimas de 4/16/64 píxeles e instancias mínimas de 0/4/16 píxeles. La mejor por Dice simétrico fue {c["min_seed_pixels"]}/{c["min_instance_pixels"]}: Dice GT {pct(c["dice_gt_macro"])}, simétrico {pct(c["dice_symmetric"])}, {c["missed"]} omisiones y {c["extra"]} extras. Estas cifras corresponden al ajuste en val, no a test; no se trasladan automáticamente de píxeles 2D a mm³.', '', 'La tabla principal y la figura conservan las instancias 2D sin este filtrado para mostrar el problema inicial. Todos los candidatos están en calibracion_instancias_2d.json. Nunca se filtra GT.']
    lines+=['', '**Regresión que debe corregirse:** el Dice del sacro bajó de 38,82 % a 23,95 %, aunque mejoraron ambos coxales y la media. La adaptación mejora la segmentación semántica global, pero todavía no resuelve la separación de fragmentos.', '', '## Qué se mantiene pendiente','',
      '- Ampliar el entrenamiento y validar en más volúmenes completos, conservando la separación por paciente.',
      '- Calibrar pérdidas y posprocesamiento en train/val, especialmente para fragmentos pequeños y superficies en contacto.',
      '- Comparar la adaptación final con SAM tras congelarla. Los resultados SAM de `revision_oct08` corresponden al modelo anterior; no se trasladan a esta variante.',
      '- La comparación de esta adaptación no abre test. Su test ya tuvo exposición exploratoria en versiones anteriores; será necesario declarar esa limitación.',
      '- Medición GPU pendiente; entorno disponible CPU. La arquitectura y pruebas no constituyen validación clínica.',
      '', '## Archivos de entrada','',
      '- `notebooks/Semana3_Sesion2_Fragmentos.ipynb`: inspección de objetivos reales, resultados y reproducción.',
      '- `scripts/entrenar_sesion2.py`, `evaluar_sesion2.py` e `inferir_sesion2_3d.py`: ejecución reproducible.',
      '- `salidas/sesion2/entrenamiento/`: configuración, historial y pesos best/last.',
      '- `tests/test_sesion2.py`: gradientes en skips, límite de ocho canales, objetivos por instancia, invariancia a IDs y separación geométrica.',
      '', 'Las salidas anteriores del equipo se preservan y se respaldan antes de actualizar código. El notebook original del profesor permanece sin modificaciones. No se consultó ni actualizó Git.']
    (ROOT/'doc/ADAPTACION_SESION2.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(ROOT/'doc/ADAPTACION_SESION2.md')

if __name__=='__main__':main()
