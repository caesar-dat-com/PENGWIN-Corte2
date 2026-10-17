# Guion de sustentación final — 25 minutos

El profesor selecciona a una persona para exponer. Los tres integrantes deben preparar el recorrido completo. El usuario confirmó que el profesor autorizó el equipo de tres integrantes. Este guion propone 15 minutos de exposición y 10 de demostración/preguntas; no certifica una presentación ya realizada.

## 0–2 min · Problema y datos

“Trabajamos con 100 tomografías de pelvis y sus máscaras anotadas. Cada tomografía es un volumen de cortes, no una fotografía. Queremos localizar sacro y ambos coxales, separar sus fragmentos y medir la distancia de cada fragmento al principal. Las etiquetas del experto permiten entrenar y comprobar el resultado.”

Mostrar la descripción de datos en `INFORME_CIERRE.md`. Explicar que la división es por paciente: 70 entrenamiento, 15 validación y 15 prueba. Se entrenó con ocho cortes por paciente; las reconstrucciones seleccionadas sí recorren todos los cortes.

## 2–4 min · Preparación y geometría

“La intensidad se expresa en HU. La ventana limita el rango mostrado, pero no convierte automáticamente la imagen en una máscara. Reorientamos imagen y etiquetas a LPS y conservamos origen, dirección y espaciado. Esto permite interpretar posiciones y distancias en milímetros. Las etiquetas se remuestrean con vecino más cercano para conservar sus identificadores.”

Mostrar el primer visualizador: MIP del CT por umbral HU. Aclarar que esta visualización no es la segmentación aprendida.

## 4–7 min · Modelo y aprendizaje

“Usamos FundidoraPC como extractor compartido, atención CBAM y tres cabezas propias: presencia anatómica, cajas y segmentación. El decodificador tiene ocho canales, conexiones de salto y supervisión auxiliar. Aprende la máscara general y señales de bordes e interiores. La salida anatómica tiene tres huesos más fondo; cada fragmento se reconstruye posteriormente dentro de su región.”

Explicar NMS como eliminación de cajas redundantes. Distinguir máscara anatómica de máscara de instancia. No describir el modelo como U-Net++ completa ni atribuirle arquitecturas que solo aparecen en el notebook del profesor.

## 7–10 min · Mejoras y comparaciones

Mostrar las tablas de calibración y ablaciones del informe. Leer las cifras del informe generado, no resultados de avances anteriores.

“Comparamos pesos de pérdida manteniendo los mismos cortes y congelando las partes que no se ajustaban. Evaluamos también con y sin CBAM, y con y sin transferencia de ImageNet en una pareja de modelos ResNet con cabezas propias. Estas últimas pruebas duran tres épocas: sirven como comparación controlada inicial, no como demostración de convergencia.”

Explicar que SAM recibe cajas predichas y sirve solo de referencia zero-shot. No produce etiquetas de entrenamiento ni integra el modelo final.

## 10–13 min · Fragmentos, medidas y errores

“La reconstrucción utiliza una política elegida en validación. No imponemos el número de fragmentos esperado ni eliminamos fragmentos de la referencia para mejorar las métricas. La distancia se calcula con EDT y espaciado físico entre centros de vóxeles de superficie. Para comparar con la referencia exigimos correspondencia tanto del fragmento como de su principal.”

Mostrar la tabla de volúmenes: GT, predichos, omitidos, extras y pares válidos. Contrastar la línea base inicial en CPU (8 cortes/paciente, 6 épocas) frente a la consolidación final en GPU Tesla T4 (30 épocas con `--modo hueso`, `9,889` cortes de entrenamiento y `2,186` de validación, mejor época 21: Macro-Dice `94.15 %` —Sacro `91.91 %`, Coxal Izq `95.09 %`, Coxal Der `95.45 %`— y `mAP@0.50 = 85.21 %`). En los tres volúmenes 3D de prueba (`004`, `006`, `010`), destacar el Macro-Dice 3D de `93.57 %`, el `dice_symmetric` de hasta `45.80 %` (`006`) y la reducción del error físico `MAE` de `0.41 mm` a `0.24 mm` gracias al control de anomalías (`0.00 mm` en `004`, y `No evaluable` en `010` por ausencia de par secundario emparejado). Mostrar los cinco peores cortes y explicar una omisión, tejido sobrante o confusión anatómica visible. Un Dice anatómico alto no equivale automáticamente a un Dice por fragmento de 85 %.

## 13–15 min · Alcance y límites

Mostrar la model card final y la latencia en CPU y GPU. Distinguir métricas de validación y prueba, y explicar la diferencia entre la política conservadora (`interfaces`, `threshold=0.7`, que logra `0` falsos positivos en `002` y `028` con `dice_sim = 46.60 %`) y la política sensible a fracturas (`interiores`, `threshold=0.35`, que recupera el `89 %–100 %` de los fragmentos y habilita los pares de distancia en mm). Reconocer exposición exploratoria previa del conjunto test, posibles fusiones de fragmentos en contacto y las ablaciones controladas de 3 épocas. Es un prototipo académico sin validación clínica.

## 15–25 min · Demostración y preguntas

1. Abrir el dashboard, seleccionar un caso y girar la MIP.
2. Recorrer cortes reales con el deslizador y señalar cajas, colores anatómicos y etiquetas de distancia.
3. Abrir la reconstrucción 3D, girarla y localizar el principal y un fragmento secundario.
4. Comparar el conteo predicho con la referencia y explicar una limitación visible.
5. Mostrar la model card y el informe descargable. Si falla la conexión, usar el dashboard local o `salidas/cierre/demo_respaldo.webm`.

El video de respaldo es un recorrido de resultados guardados, no una grabación de inferencia en vivo ni la sustentación completa. Reservar aproximadamente cuatro minutos para preguntas.
