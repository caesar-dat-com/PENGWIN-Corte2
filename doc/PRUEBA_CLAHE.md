# Prueba CLAHE — 8 de octubre de 2026

Resultado: no desplegar esta variante. La versión actual sin CLAHE conserva mejores resultados.

Se compararon los mismos 120 cortes de 15 pacientes de validación. La adaptación utiliza 560 cortes reales de 70 pacientes de entrenamiento. No se usó test. CLAHE se aplica después de la ventana HU de nivel 400 y ancho 1800, con clipLimit=2 y cuadrícula 8×8 sobre la imagen de 256×256, cuantizada a 8 bits. Las tomografías originales permanecen intactas.

| Variante | mAP50 cajas | Dice semántico macro | Píxeles falsos positivos de fondo | Sensibilidad al hueso objetivo |
|---|---:|---:|---:|---:|
| Actual, sin CLAHE | 80.85% | 68.46% | 77793 | 85.77% |
| CLAHE sin adaptar | 30.03% | 32.28% | 445727 | 79.66% |
| Control sin CLAHE, 3 épocas | 80.85% | 68.09% | 87622 | 88.75% |
| CLAHE, segmentador adaptado 3 épocas | 30.03% | 56.77% | 119930 | 80.13% |

## Alcance y límites

Las cajas se recalcularon en cada variante: no se reutilizaron cajas obtenidas sobre las imágenes sin CLAHE. La prueba inicial utiliza los mismos pesos. La segunda adapta únicamente el decoder durante tres épocas con AdamW 1e-4, lote ocho, semilla 42 y las mismas pérdidas del control. Se reutilizó el control de tres épocas ya ejecutado con esas condiciones. Encoder, detector y RPN permanecieron congelados; se comprobó igualdad exacta de parámetros y buffers externos al decoder. Por ello la métrica de cajas con CLAHE es idéntica antes y después de adaptar el decoder.

No es una prueba de entrenamiento completo con CLAHE ni de todos sus parámetros posibles. Demuestra que esta configuración, añadida al sistema actual y con adaptación limitada al segmentador, no mejora el resultado. Un ensayo completo de encoder/detector/RPN requeriría otro entrenamiento y nueva validación. No se atribuye causalmente toda la caída a CLAHE como técnica universal: hay cambio de distribución respecto al entrenamiento del detector.

Las métricas de máscaras son semánticas, no de fragmentos individuales. Se seleccionó la mejor época del decoder en validación; estos datos no son una evaluación independiente. No se ejecutó reconstrucción volumétrica ni se sustituyó el visor debido al deterioro ya observado. 38 pruebas automatizadas aprobadas.

Los ensayos anteriores de negativos difíciles con pesos 0.5 y 0.1 tampoco cumplieron todas las salvaguardas de sensibilidad y Dice regional; sus resultados permanecen documentados y el modelo del visor se conserva.
