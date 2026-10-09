# Revisión local de orientación y cajas — 8 de octubre de 2026

La MIP recibe matrices con superior en la primera fila. Se corrige el eje vertical de Plotly con autorange reversed; la geometría LPS del volumen no se modifica. Una prueba con dos puntos de intensidades diferentes comprueba la orientación en cuatro ángulos.

El visor utiliza ahora la variante propia inspirada en RPN sobre cuadrícula, con anclas derivadas de train, NMS propio y respaldo del detector anterior cuando falta una propuesta confiable. Se selecciona una caja macro por región anatómica; esto no fija el número de fragmentos. No se usa Faster R-CNN preentrenado ni anotaciones para inferir.

Política seleccionada en validación: confianza RPN 0.10, respaldo 0.25, sin ampliar las cajas. Las máscaras se recalculan con las cajas nuevas; posteriormente se reconstruyen instancias y distancias con la política anterior congelada. Los resultados anteriores permanecen en salidas/cierre. Los nuevos se guardan en salidas/cajas_revision/volumenes.

En 120 cortes de validación: mAP50 0.808523, mAP50:95 0.405113, IoU media a confianza 0.25 0.669929. La cobertura media de máscara es 0.919721; queda una región sin detectar. Son resultados usados para seleccionar la variante, no una evaluación independiente. Una caja por región elimina duplicados por construcción; no garantiza que cada caja esté bien localizada.

La demostración revisada comprende únicamente los volúmenes de validación 002, 012 y 028. Los resultados históricos de prueba no se atribuyen a la RPN. El informe y video anteriores están identificados como históricos en el visor.

Referencia conceptual: Ren et al., Faster R-CNN (2015), https://arxiv.org/abs/1506.01497. La implementación propia es una propuesta por anclas con clase; no reproduce toda la arquitectura de dos etapas del artículo.

## Comprobación volumétrica

906 cortes contiguos verificados: cero cajas duplicadas por clase, máscaras restringidas a las cajas correspondientes, geometría LPS alineada entre CT, máscaras, instancias y referencia. 32 pruebas automatizadas aprobadas.

| Caso | Instancias predichas | Sobrantes | Omitidas | Dice de fragmentos |
|---|---:|---:|---:|---:|
| 002 | 11 | 8 | 3 | 0.3653 |
| 012 | 21 | 17 | 1 | 0.3150 |
| 028 | 13 | 10 | 4 | 0.2881 |

No hay pares de fragmentos y principal válidos para evaluar el error de distancia. Persisten errores de segmentación.

## Revisión de fragmentos y supuesto fondo extracorporal

Solicitud: mejorar fragmentos fusionados, sobrantes y omitidos; posteriormente revisar fragmentos aparentemente fuera del cuerpo.

Se construyó una envolvente de CT original remuestreado a la geometría de inferencia: HU > -500, apertura axial de 2 mm, rellenado de cavidades, componente corporal 3D principal y margen físico de 3 mm. El filtro no consulta GT. La evaluación independiente del filtro en estos tres volúmenes encontró cero vóxeles predichos externos y cero vóxeles GT excluidos. Por tanto, no elimina los falsos positivos actuales: estar dentro del cuerpo no implica pertenecer a los huesos pélvicos objetivo. No se debe presentar este filtro como una mejora medida de segmentación.

Se compararon cinco alternativas de separación sobre predicciones fijas: interfaces con umbrales 0.3 y 0.7, semillas mínimas 200 mm³, interiores y componentes conexos. El mínimo de volumen se mantuvo en 500 mm³ y nunca se eliminó GT. Se seleccionó interfaces 0.7, semilla mínima 50 mm³: extras totales 35 → 32, omitidos 8 → 8, emparejamientos IoU≥0.5 4 → 4. Mejora marginal, especialmente en 012; el caso 002 conserva ocho extras. En 028 el Dice GT baja de 0.288070 a 0.287760, dentro de la tolerancia prefijada de 0.005 por caso. No se afirma mejora en cada indicador.

Los resultados nuevos permanecen en salidas/fragmentos_revision/volumenes; los anteriores se conservan. La comparación anatómica blanca del visor usa anotaciones exclusivamente para evaluar visualmente. 35 pruebas automatizadas aprobadas. No se usó test ni se modificó Git/OneDrive.

Pendiente para corregir la confusión anatómica: entrenamiento con ejemplos negativos de huesos y tejidos fuera de la pelvis objetivo, incluyendo cortes sin regiones objetivo, y contexto entre cortes. No eliminar automáticamente una isla solo por ser pequeña o distante: podría ser un fragmento verdadero. Las distancias siguen sin pares válidos suficientes para acreditar precisión.
