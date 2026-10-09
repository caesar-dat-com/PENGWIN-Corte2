# PENGWIN: integración y evaluación del pipeline local

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
| inicial | 60.45% | 36.89% | 0 |
| control | 64.10% | 43.36% | 4 |
| precision | 64.95% | 46.74% | 4 |
| contorno | 66.72% | 52.89% | 4 |

La selección de decoder fue **contorno**. Ajuste de cajas aceptado: **False**. Se evita atribuir toda mejora a un operador aislado. El refinamiento bilateral de la revisión anterior se deja apagado en esta fase: su efecto fue marginal y se prioriza corregir la predicción aprendida.

Comparación visual en tres posiciones fijas (25/50/75 %) del caso 002 de validación. La imagen anterior incluye su refinamiento seleccionado; las tablas de calibración usan la comparación controlada sin ese refinamiento. Se observan contornos más ajustados en algunas zonas, pero también recortes por cajas y confusiones anatómicas que persisten.

![Comparación visual de validación](../salidas/cierre/comparacion_visual_val.png)

## Ablaciones requeridas

Tres épocas por ejecución, misma partición, inicialización de cabezas comprobada por hash dentro de cada par, LR 0,0003 y lote 8. Fundidora con/sin CBAM evalúa atención. ResNet18 con/sin pesos ImageNet evalúa transferencia dentro de la misma arquitectura. Las cabezas se entrenan desde cero en ambos ResNet. No se interpreta la diferencia Fundidora/ResNet como efecto de transferencia.

| Variante | Dice semántico val | mAP50 val | F1 clasificación val |
|---|---:|---:|---:|
| fundidora_cbam | 14.21% | 39.92% | 91.34% |
| fundidora_sin_cbam | 14.38% | 31.73% | 84.92% |
| resnet_sin_transfer | 49.08% | 45.74% | 92.49% |
| resnet_transfer | 33.82% | 28.19% | 95.72% |

Es un estudio corto y controlado, no una demostración de convergencia. CBAM se mantiene en la arquitectura final porque es un requisito; los resultados no se fuerzan a favorecerlo.

## Resultados de prueba y SAM

| Métrica | Resultado | Objetivo sugerido |
|---|---:|---:|
| F1 clasificación por corte | 91.23% | 85 % |
| AUC clasificación por corte | 97.98% | 85 % |
| IoU de cajas incluyendo omisiones | 56.21% | 65 % |
| mAP50 | 56.14% | 65 % |
| mAP50:95 | 18.20% | 40 % |

Clasificación mide presencia anatómica por corte, no exactitud de cada fragmento. La definición de IoU de cajas incluye GT omitidos como cero. Los objetivos del enunciado son sugeridos; no se afirman alcanzados cuando no lo están.

SAM ViT-B recibe exclusivamente cajas predichas tras NMS; no recibe cajas ni máscaras GT, no se ajusta y no participa en el pipeline final.

| 15 cortes centrales test | Dice anatómico | IoU anatómico |
|---|---:|---:|
| Modelo propio | 79.47% | 66.93% |
| SAM zero-shot | 85.55% | 75.97% |

Estas cifras anatómicas no son el Dice por fragmento exigido. La comparación también guarda instancias 2D con asignación 1:1, omisiones y extras en `comparacion_test/metricas.json`; sus filtros se expresan en píxeles 2D y no se confunden con mm³. SAM no está diseñado aquí para separar todos los fragmentos que comparten una caja anatómica.

![Matrices de clasificación por presencia](../salidas/cierre/matriz_clasificacion.png)

## Fragmentos 3D y distancias

Se compararon siete políticas sobre tres volúmenes val: interiores aprendidos, componentes conectados, interfaces aprendidas con tres umbrales y dos filtros de volumen adicionales. Se selecciona por Dice simétrico que penaliza extras y omisiones; no se fuerza un máximo de diez predicciones ni se borra GT. Política congelada: `{'method': 'interfaces', 'threshold': 0.5, 'seed_mm3': 50.0, 'min_volume_mm3': 500.0}`. Un umbral mínimo de volumen puede descartar fragmentos verdaderos pequeños; esa consecuencia cuenta en las métricas.

Resultados sobre validación, utilizada para seleccionar la política:

| Caso val | GT | Predichos | Omitidos | Extras | Dice GT | Dice simétrico |
|---|---:|---:|---:|---:|---:|---:|
| 002 | 6 | 20 | 3 | 17 | 36.06% | 9.41% |
| 012 | 5 | 26 | 1 | 22 | 28.28% | 5.24% |
| 028 | 7 | 22 | 4 | 19 | 27.69% | 7.45% |

En el caso 002, la versión anterior produjo 96 instancias para seis fragmentos GT, con dos omitidos y 92 extras; Dice GT 32,30 % y Dice simétrico 1,98 %. La nueva versión reduce extras pero puede aumentar omisiones. No es una mejora uniforme y este caso interviene en la selección.

| Caso test | GT | Predichos | Omitidos | Extras | Dice GT | IoU GT | Dice simétrico | Pares distancia | MAE mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 004 | 6 | 16 | 2 | 12 | 35.18% | 27.18% | 11.73% | 0 | No evaluable |
| 006 | 7 | 13 | 2 | 8 | 43.82% | 32.66% | 20.45% | 0 | No evaluable |
| 010 | 5 | 14 | 2 | 11 | 31.73% | 21.84% | 9.91% | 0 | No evaluable |

Dice GT promedia los fragmentos de referencia, incluyendo omitidos como cero; no penaliza directamente cada predicción extra. Dice simétrico incluye además esas predicciones sin correspondencia como cero y revela la sobresegmentación. El objetivo sugerido de Dice por fragmento 85 % / IoU 70 % no debe confundirse con el Dice anatómico.

Pares de distancia válidos totales: **0**. MAE agregado: **no evaluable**. Solo se comparan distancias si fragmento y principal se emparejan correctamente con IoU ≥ 0,5. Un valor no evaluable nunca se transforma en cero. El principal predicho es la instancia mayor de la región; el GT usa IDs 1/11/21. La distancia se calcula mediante `distance_transform_edt` con espaciado físico, entre centros de vóxeles superficiales; es una aproximación discreta de borde a borde. Contacto 26 se informa aparte. La reducción XY a 256 conserva campo físico y todos los cortes Z, pero puede perder fragmentos pequeños.

## Cinco peores casos y análisis de errores

Se seleccionan automáticamente los cinco peores **cortes centrales** entre los 15 evaluados con SAM, por Dice anatómico. No representan necesariamente los cinco peores volúmenes completos. Las hipótesis se basan en falsos positivos, omisiones y confusión entre regiones, sin atribuir diagnósticos clínicos.

- 095_128: Predomina tejido sobrante; revisar especificidad y límites de las cajas.
- 079_134: Predomina tejido sobrante; revisar especificidad y límites de las cajas. Hay confusión entre regiones anatómicas.
- 078_194: Predomina tejido sobrante; revisar especificidad y límites de las cajas.
- 091_136: Predomina tejido sobrante; revisar especificidad y límites de las cajas.
- 011_156: Predomina tejido sobrante; revisar especificidad y límites de las cajas.

![Cinco peores cortes](../salidas/cierre/cinco_peores.png)

## Visualización, reproducción y cómputo

El dashboard integra MIP del CT por umbral HU sin modelo, recorrido de todos los cortes con cajas/máscaras y distancias sobre la imagen, y malla 3D con marching cubes y Plotly Mesh3d, coloreada por hueso y etiquetada por fragmento. La malla se simplifica solo para visualizar; las distancias usan el volumen. Los casos mostrados son resultados ya calculados, no inferencia interactiva sobre archivos nuevos. Es accesible desde un navegador y tiene disposición adaptable a pantalla estrecha.

CPU: 34.42 ms/corte de media; p95 48.22 ms, lote 1, cinco warmups y 24 cortes reales. Incluye forward, NMS, restricción de máscaras y transferencia a CPU; excluye IO, resize y reconstrucción 3D. GPU: **unavailable**. El entorno instalado es CPU y no se encontró `nvidia-smi`; no se declara que el equipo carezca físicamente de GPU. Falta medir en un entorno CUDA funcional. La prueba está en `scripts/medir_latencia.py`.

Las rutas y comandos están en README, los pesos conservan hashes y los paquetes exactos están en `requirements-reproducible.txt`. GitHub, protección de main y revisiones por integrante no se verificaron ni modificaron por instrucción del usuario. El túnel de Cloudflare es temporal y solo queda cumplido cuando exista una URL comprobada; consultar el registro de despliegue.

## Limitaciones y uso responsable

Uso exclusivamente académico. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales. Persisten errores de segmentación, fusiones de fragmentos en contacto, ruido y sensibilidad a cortes poco representados. No se garantiza alcanzar Dice de fragmento ≥ 85 % ni IoU ≥ 70 %. El muestreo pequeño, el test previamente explorado y las ablaciones de tres épocas limitan generalización. Las distancias deben interpretarse junto con cobertura de emparejamiento y calidad de segmentación, nunca de forma aislada.

## Referencias y procedencia

- Enunciado local `Proyecto_Corte2_PENGWIN.docx`, secciones 3–7.
- Notebook de Sesión 2 del profesor, adaptado en la revisión anterior; no ejecutado como pipeline final de YOLO/Mask R-CNN.
- Pesos oficiales de backbone ImageNet: https://download.pytorch.org/models/resnet18-f37072fd.pth (SHA256 registrado en ablaciones).
- SimpleITK DICOMOrient: https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1DICOMOrientImageFilter.html
- Cloudflare Quick Tunnels: https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/
