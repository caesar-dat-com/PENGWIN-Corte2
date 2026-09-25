# IA_USAGE — registro de uso de IA generativa

| Fecha | Herramienta | Semana | Prompt exacto | Qué se obtuvo | Modificaciones manuales del equipo |
|---|---|---|---|---|---|
| 2026-09-23 | Claude Opus 5.5 vía Claude Code | 1 (sem. 8) | «/home/caesar/Downloads/Proyecto_Corte2_PENGWIN.docx realiza lo siguiente ya cree el drive donde ya lo inicie https://drive.google.com/drive/folders/1nMLAmQNrnX9keImtL_dM550PliNKbmbq usa el material de la clase de analitica de datos que esta en el obsidian» + «solo debemos hacer la semana 01» | Paquete `pengwin/` (carga .mha + reorientación LPS, ventaneo HU, EDA de fragmentos con distancia EDT en mm, splits estratificados por paciente, visualizador 1 MIP rotatorio) y notebook `notebooks/Semana1_Datos_EDA_MIP.ipynb` | Ver detalle abajo |

## Modificaciones manuales del equipo

El código generado corría, pero no salió bien de entrada. Lo que cambiamos
nosotros, en orden de importancia:

1. **Reorientación a LPS.** La primera versión cargaba los `.mha` tal como
   venían. Al comparar casos vimos que no comparten dirección: el 001 está en
   RAI y el 002 en LPI. Sin reorientar, la mitad de los volúmenes queda espejada
   y un modelo 2D aprendería la lateralidad invertida — llamaría coxal izquierdo
   al derecho. Agregamos `sitk.DICOMOrient` a LPS en la carga.

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

- **Ignoró la orientación de los volúmenes.** El error más costoso: habría
  invertido la lateralidad en la mitad del dataset sin que ninguna métrica de
  entrenamiento lo delatara. Solo se detecta comparando headers entre casos.
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
