# Guion de sustentación — Semana 9 (Avance 2)

Revisado el 8 de octubre de 2026. Este guion explica la demostración original de detección; los experimentos nuevos están separados en `salidas/revision_oct08/`. No atribuir sus resultados al video o al cuaderno anterior.

Equipo: Yesenia Díaz, Juan Pablo Maya y César Reyes. Tiempo aproximado: 6 minutos.

## 1. Yesenia — datos y preparación (0:00–2:00)

Mostrar: portada del cuaderno de semana 9 y `salidas/revision_oct08/auditoria_areas.png`.

“Presentamos el segundo avance de PENGWIN. Trabajamos con tomografías de pelvis y sus máscaras anotadas. Cada tomografía es un volumen formado por cortes. Las etiquetas permiten reconocer el sacro, el coxal izquierdo y el derecho, y distinguir fragmentos dentro de cada región.

Para la detección generamos una caja por región presente en cada corte. Una caja puede abarcar varios fragmentos: detectar una región todavía no significa separar todos sus fragmentos.

Leemos la geometría física y reorientamos imagen y máscara de manera coherente. Aplicamos una ventana de nivel 400 y ancho 1800 HU: los valores entre −500 y 1300 se normalizan al intervalo de cero a uno. Usamos números de punto flotante para el procesamiento.

La revisión encontró que el inventario anterior calculaba porcentajes supuestos, en lugar de contar directamente las máscaras. Ahora sí contamos las 100 máscaras: hay 62.738 apariciones de regiones por corte. Un umbral de 15 píxeles excluiría 255 cajas, el 0,41 %. No tenemos evidencia para llamar ruido a esas anotaciones. Por ello conservamos todas las regiones anotadas por defecto.

Un fragmento grande en tres dimensiones puede ocupar muy pocos píxeles en uno de sus cortes extremos. Por eso, su volumen total no permite justificar eliminar una anotación pequeña en dos dimensiones.”

## 2. Juan Pablo — arquitectura (2:00–4:00)

Mostrar: esquema y secciones de backbone, CBAM, detector y NMS del cuaderno.

“La red toma cortes de 256 por 256 píxeles. Repetimos la imagen gris en tres canales: eso no agrega información de color. El backbone FundidoraPC extrae características mediante cuatro etapas residuales, hasta obtener 256 canales en una cuadrícula de 16 por 16.

Después aplicamos CBAM, que aprende a ponderar canales y posiciones espaciales. Conservamos la convolución espacial de 9 por 9 de nuestra implementación. Su ventana contiene más celdas que una de 7 por 7, pero demostrar que mejora el resultado requiere comparar ambas configuraciones; no basta con aumentar su tamaño.

La cabeza de detección produce ocho valores por celda: presencia de objeto, cuatro parámetros de caja y tres clases anatómicas. La cabeza de clasificación indica qué regiones están presentes en el corte.

Finalmente, NMS elimina cajas redundantes según su superposición, por separado para cada clase. Está implementado en el proyecto. Una limitación de esta cuadrícula es que solo admite un objetivo por celda. La versión revisada detecta una colisión y detiene el entrenamiento, en lugar de sobrescribir una anotación sin avisar.

El avance posterior incorpora un decodificador con ocho canales internos y una salida semántica de cuatro clases. Los ocho canales de la cabeza de cajas y los ocho del decodificador son dimensiones de módulos diferentes.”

## 3. César — demostración, límites y siguiente avance (4:00–6:00)

Mostrar: salidas guardadas del cuaderno, curva de pérdida y comparación de cajas reales.

“Combinamos pérdidas de presencia de objeto, coordenadas de cajas, clase de la caja y presencia de regiones en el corte. La Focal Loss, con gamma 0,5, reduce la contribución de ejemplos fáciles. Su función aquí es entrenar la detección; no mide directamente la calidad de los bordes de una fractura.

La demostración original utiliza cuatro cortes reales del caso 001: 134, 220, 235 y 250. Se entrena y se evalúa sobre esos mismos cortes para comprobar que la red puede aprender un lote pequeño. Esto se denomina overfit intencional.

El cuaderno guardado muestra una reducción de pérdida aproximada de 2,6496 a 0,0527. La cifra pertenece a esa ejecución guardada; si repetimos el experimento debemos presentar el registro de la nueva ejecución. Una pérdida pequeña y cajas visualmente próximas no prueban precisión milimétrica ni buen rendimiento en otros pacientes.

El siguiente paso es entrenar con pacientes de entrenamiento y seleccionar el modelo con pacientes de validación, manteniendo separado el conjunto de prueba. También debemos separar fragmentos, medir la distancia de cada fragmento al principal del mismo hueso en tres dimensiones, y comparar contra las anotaciones.

Las etiquetas individuales de fragmentos son identificadores dentro de cada caso; no son quince clases anatómicas. Para el contraste con SAM usaremos cajas predichas por nuestro detector y documentaremos tanto aciertos como omisiones. Los resultados nuevos deben consultarse en el informe de revisión, sin mezclarlos con esta demostración inicial.”

## Antes de grabar

- Abrir las figuras indicadas y comprobar que corresponden a la ejecución que se explica.
- No usar la antigua figura `inventario_ruido_componentes.png`: sus porcentajes eran supuestos.
- Mantener separadas las expresiones “detección de regiones”, “segmentación semántica” y “segmentación de fragmentos”.
- No afirmar validación clínica, ausencia de errores o generalización a partir del overfit.
