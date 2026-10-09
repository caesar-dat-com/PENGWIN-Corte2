# Model card del modelo final seleccionado

Uso exclusivamente académico. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

Arquitectura: FundidoraPC + CBAM, tres cabezas propias, decoder con skips y ocho canales. Backbone inicialmente desde cero, sin pesos de SAM ni de detectores externos. Entrenamiento y selección en 70/15 pacientes; resultados de prueba limitados a los cortes y volúmenes descritos en el informe.

Checkpoint: `salidas\cierre\ajuste\contorno\best.pth`. SHA256: `2a5e9f1daeec2328da2b78f263aa8edc1d10aa62593bfe5f18f56f494fc7297d`.

Clasificación test: F1 91.23%, AUC 97.98%. Detección test: mAP50 56.14%, mAP50:95 18.20%. Dice anatómico en 15 cortes centrales: 79.47%; no equivale a Dice por fragmento. Resultados volumétricos y distancias en `INFORME_CIERRE.md`.

Política de instancias: `{'method': 'interfaces', 'threshold': 0.5, 'seed_mm3': 50.0, 'min_volume_mm3': 500.0}`. Puede omitir fragmentos pequeños y fusionar fragmentos en contacto; no forzar resultados a la taxonomía esperada. Distancia EDT entre centros de vóxeles superficiales, con spacing, y comparación válida solo con correspondencia de fragmento y principal.

No usar para decisiones clínicas. Test previamente explorado, pocos cortes de entrenamiento, rendimiento insuficiente en casos difíciles, GPU pendiente de medir. Las alertas son control de calidad y no diagnóstico. La reproducción usa la configuración y los hashes guardados; no modificar el protocolo tras observar test.
