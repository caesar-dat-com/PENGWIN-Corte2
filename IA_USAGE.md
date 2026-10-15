# IA_USAGE — registro de uso de IA generativa

| Fecha | Herramienta | Semana | Prompt exacto | Qué se obtuvo | Modificaciones manuales del equipo |
|---|---|---|---|---|---|
| 2026-09-23 | Claude Opus 5.5 vía Claude Code | 1 (sem. 8) | «/home/caesar/Downloads/Proyecto_Corte2_PENGWIN.docx realiza lo siguiente ya cree el drive donde ya lo inicie https://drive.google.com/drive/folders/1nMLAmQNrnX9keImtL_dM550PliNKbmbq usa el material de la clase de analitica de datos que esta en el obsidian» + «solo debemos hacer la semana 01» | Paquete `pengwin/` (carga .mha + reorientación LPS, ventaneo HU, EDA de fragmentos con distancia EDT en mm, splits estratificados por paciente, visualizador 1 MIP rotatorio) y notebook `notebooks/Semana1_Datos_EDA_MIP.ipynb` | Ver detalle abajo |

## Modificaciones manuales del equipo

El código generado corría, pero no salió bien de entrada. Lo que cambiamos
nosotros, en orden de importancia:

1. **Reorientación a LPS.** La primera versión cargaba los `.mha` tal como
   venían. Al comparar casos vimos que no comparten dirección: la auditoría actual registra el 001 en LPS y el 002 en RAS.
   La afirmación anterior sobre RAI/LPI era incorrecta. En los 100 headers hay
   66 LPS y 34 RAS. La carga usa `sitk.DICOMOrient` para estandarizar a LPS
   sin cambiar los puntos físicos; no basta con voltear una matriz visualmente.

2. **Espaciado del header en todas las medidas.** El cálculo inicial de volumen
   trataba los vóxeles como isotrópicos. No lo son: el espaciado varía por caso
   (0,78–1,0 mm según el eje). Cambiamos volumen y distancias para leer
   `pixdim` del header, y pasamos `sampling=spacing` a
   `distance_transform_edt`. Sin eso, las distancias en "mm" eran en vóxeles
   disfrazados.

3. **Split por caso completo, no por corte.** La propuesta inicial repartía
   cortes. Cortes vecinos del mismo CT son casi idénticos, así que eso mete en
   test datos prácticamente vistos en train y las métricas salen infladas.
   Rehicimos el split con el caso como unidad, estratificado por número de
   huesos fracturados y con semilla fija, y congelamos `splits/splits.json` en
   el repositorio.

4. **Descartamos el umbral de 200 HU como segmentación.** Medimos los
   percentiles de HU dentro de la máscara del experto y encontramos que en el
   caso 001 el **54 %** del hueso anotado está por debajo de 200 HU (41 % en el
   002, 15 % en el 003). Umbralizar ahí borraría medio sacro. Dejamos el umbral
   solo para el MIP —donde funciona porque la proyección se queda con la
   cortical— y le pasamos a la red una ventana ancha (L=400, W=1800).

5. **QC de headers antes de cargar vóxeles.** No estaba. Lo agregamos para
   verificar tamaño, espaciado, origen, dirección, `scl_slope = 1` y
   correspondencia `ElementSpacing` ↔ `pixdim` en los 100 casos, antes de gastar
   10 minutos de cómputo.

## Análisis crítico

### Aciertos

- Ahorró el trabajo mecánico: leer `.mha` con SimpleITK, armar el EDT con
  `scipy`, montar el MIP rotatorio con slider en HTML autocontenido. Eso desde
  cero nos habría tomado varios días.
- La estructura modular (`io.py`, `eda.py`, `splits.py`, `viz_mip.py`) salió
  limpia y nos sirve para las semanas siguientes sin reescribirla.
- Cacheo de resultados: si `eda_casos.csv` existe, no recalcula. Detalle pequeño
  pero nos ahorró repetir 10 minutos cada vez que reabríamos el notebook.

### Errores detectados y corregidos

- **Ignoró la orientación de los volúmenes.** La auditoría de headers justifica estandarizar
  orientación antes de usar las matrices. La proporción y los códigos de
  orientación de la descripción original fueron corregidos arriba.
- **Trató los vóxeles como isotrópicos.** Volúmenes y distancias mal calculados
  aunque el código corriera sin error.
- **Propuso split por corte**, que es fuga de datos entre train y test.
- **Asumió 200 HU como umbral válido de hueso** por ser el valor clásico, sin
  medirlo en este dataset. Al medirlo, resultó inservible para segmentar.

El patrón es claro: los errores no eran de sintaxis ni de librería —el código
corría— sino de supuestos sobre los datos que la IA no verificó. Ninguno habría
aparecido sin abrir los archivos y medir.

### Limitaciones

- **Solo 100 casos** de entrenamiento, y de ellos únicamente 16 tienen los tres
  huesos fracturados. El test queda con 15 casos: cualquier métrica sobre él va
  a tener varianza alta.
- **Metal en el dataset.** El caso 003 alcanza 16.709 HU, muy por encima del
  rango del hueso. Es material quirúrgico o artefacto, y va a interferir en
  segmentación. Queda identificado pero no tratado esta semana.
- **Desbalance 2D.** Contando cortes y no casos, los coxales aparecen bastante
  más que el sacro. Si no se compensa en el muestreo, el modelo va a tender a
  ignorar el sacro.
- **`dist_mm_gt` se mide sobre el ground truth**, no sobre predicciones. Es la
  referencia para la semana 10, pero todavía no valida nada.
- **Sin validación clínica.** El ground truth es el del challenge; no tenemos
  criterio médico para juzgar si una anotación está bien hecha.


## Revisión local — 8 de octubre de 2026

Herramienta: Codex. Solicitud del usuario: «adelante», autorizando continuar las mejoras de la versión local revisada. No se consultó ni actualizó Git.

La asistencia automatizada se utilizó para sustituir porcentajes supuestos por conteos de las 100 máscaras, conservar GT pequeño, preparar muestreo separado por paciente, incorporar una cabeza auxiliar de bordes al decodificador existente, implementar evaluación de fragmentos 3D y comparación SAM, y corregir documentación. Se reutilizaron utilidades locales de métricas e instancias revisadas; se mantuvo la arquitectura de la nueva versión del equipo (CBAM9 y decoder8).

Se verificaron invariancia a renumerar fragmentos, separación de instancias, espaciado físico, preservación de GT, gradientes y métricas. Los resultados concretos y limitaciones están en `doc/REVISION_OCT08.md`; las cifras del experimento se leen de los registros, no se estiman. El código y la interpretación deben ser revisados por el equipo antes de la entrega. La asistencia no constituye validación clínica ni autoría íntegramente manual del equipo.


### Adaptación autorizada del notebook de Sesión 2

Tras la indicación «adelante», se adaptaron conceptos de `FundidoraUNet`, `FundidoraUNetPP`, supervisión profunda y máscaras individuales del notebook proporcionado por el profesor. La adaptación utiliza skips aditivas, ocho canales y cabezas auxiliares de bordes e interiores. No copia los experimentos de YOLO/Mask R-CNN como arquitectura final ni sus resultados. Los IDs de fragmento se conservan y no se tratan como clases anatómicas globales. El código, pruebas y documentación se prepararon con asistencia de Codex; el equipo debe revisar y poder explicar las decisiones.


### Máscara macro, LPS y Sobel — sesión con el profesor

A solicitud explícita del usuario, Codex revisó el artículo de MethodsX (DOI 10.1016/j.mex.2024.103073) y adaptó sus ideas pertinentes de consistencia geométrica, sin trasladar correcciones craneales a pelvis. Se implementaron pérdida macro y contorno Sobel, refinamiento local, validaciones LPS y alertas heurísticas de calidad. Se compararon entrenamiento adicional de control y candidato con la misma inicialización y partición por paciente. Las métricas y límites se leen de los resultados guardados; no se atribuyen a test ni al modelo anterior. Se preservaron los experimentos originales y no se usó Git. El equipo debe comprender y revisar el código y su interpretación antes de entregarlo.


## Integración y cierre local — 8 de octubre de 2026

Herramienta: Codex, agente basado en GPT-6. No se declara una revisión exacta del modelo que no fue proporcionada por la sesión. Prompts textuales que motivan esta intervención: «se cumple con los requerimientos del proyecto ?» y «realiza las mejorar necesarias». Confirmación posterior del usuario sobre el tamaño del equipo: «Sí, autorizó tres integrantes».

Sugerencias y trabajo generado: calibrar pérdidas de máscara/contorno en validación; ajustar localización de cajas sin alterar el decoder; comparar CBAM y transferencia mediante parejas controladas; evaluar instancias y distancias en volúmenes reales; repetir SAM con protocolo fijo; integrar MIP, cortes y malla 3D; generar informe, model cards, guion y video de respaldo. Se conservó la restricción local de decoder de ocho canales y no se utilizó Git.

La asistencia automatizada escribió e integró código, ejecutó entrenamientos y pruebas y generó documentación a partir de los registros. Las métricas exactas se encuentran en `doc/INFORME_CIERRE.md` y `salidas/cierre/`; no se atribuyen a ejecuciones históricas. Las modificaciones manuales de código del equipo durante esta intervención **no están documentadas**: no deben presentarse estos cambios como programación íntegramente manual. El usuario autorizó el trabajo local y confirmó la excepción del equipo de tres.

Análisis crítico: mejorar Dice anatómico no demuestra separación correcta de fragmentos. Se retienen GT pequeño, omisiones y falsos positivos, y se informa la cobertura de emparejamientos antes de interpretar el MAE. Los hiperparámetros se eligieron con validación; el test ya había sido explorado antes y no se presenta como completamente independiente. Las ablaciones cortas no demuestran convergencia. La ejecución CPU no sustituye una medición GPU. Las alertas de calidad no son detección clínica de anomalías. El informe registra resultados favorables y limitaciones; el equipo debe revisar y poder explicar ambos.


## Revisión local de orientación y RPN — 2026-10-08
Solicitud del usuario: «corrige», referida a MIP invertida y cajas superpuestas. Se corrigió la convención de pantalla, se integró la variante RPN propia seleccionada en validación y se recalcularon máscaras e instancias. Se añadieron marcadores anatómicos y una prueba de orientación con un volumen sintético exclusivamente de prueba, no de entrenamiento. Los resultados de validación no acreditan rendimiento independiente ni ausencia de errores. No se modificó Git ni la carpeta original de OneDrive.


## Separación y envolvente corporal
Usuario: «adelante» y «hay fragmentos fuera de lo que corresponde al cuerpo, como lo corregimos?». Se compararon cinco políticas en tres volúmenes val, se implementó y auditó envolvente corporal basada exclusivamente en CT, y se añadió contorno de referencia para inspección. La auditoría contradijo la hipótesis inicial de predicciones extracorporales: ninguna estaba fuera de la envolvente con margen. Se informó esa limitación; no se atribuyó a la envolvente la reducción de extras conseguida por separación.


## Entrenamiento de negativos difíciles
Usuario: «adelante», autorizando mejorar la identificación anatómica. Se preparó comparación controlada de continuación de entrenamiento frente a pérdida adicional sobre fondo anotado difícil, manteniendo RPN y encoder congelados. No se usaron máscaras de validación para construir ejemplos de entrenamiento. Las pruebas verifican que la penalización excluye el hueso anotado y genera gradiente hacia fondo para un falso positivo. Los resultados se seleccionan en val y requieren revisión volumétrica antes del cambio del visor.


## Ensayo CLAHE solicitado
Usuario: «has la prueba». Se comparó inferencia con y sin CLAHE y adaptación de tres épocas del decoder con CLAHE consistente en train/val, reutilizando control idéntico sin CLAHE. Se recalcularon cajas para evitar evaluación con cajas de otra entrada. No se reentrenaron encoder/detector/RPN; esta limitación se documentó. El resultado fue peor y no se desplegó. Ver doc/PRUEBA_CLAHE.md y salidas/clahe_revision/decision.json.
