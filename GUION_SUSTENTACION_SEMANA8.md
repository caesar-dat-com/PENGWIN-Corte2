# Guion de Sustentación — Semana 8 (Semana 1 del Proyecto)
## Proyecto Integrador: Sistema PENGWIN
**Analítica de Datos · UAO 2026-2 · Prof. Carlos A. Ferro**  
**Equipo:** Yesenia Díaz · Juan Pablo Maya · César Reyes  
**Límite de tiempo:** Máximo 5:00 minutos | **Tiempo objetivo calibrado:** 4:20 minutos (margen de 40 s)

---

## 📋 Distribución de Roles y Diapositivas

| Integrante | Diapositivas | Eje Temático | Tiempo Asignado |
|---|:---:|---|:---:|
| **1. Yesenia Díaz** | **1, 2 y 3** | Portada, advertencia académica, taxonomía LPS, métrica de separación en mm (GT) y **Gráfica 1** (distribución de fragmentos). | **1 min 25 s** (0:00 – 1:25) |
| **2. Juan Pablo Maya** | **4, 5 y 6** | Desbalance de cortes axiales 2D, **Gráfica 2** (análisis métrico de separación) e intensidad radiológica HU del hueso trabecular. | **1 min 25 s** (1:25 – 2:50) |
| **3. César Reyes** | **7, 8 y 9** | Partición estratificada por paciente (`splits.json`), muestra visual del **Caso 001** (ventaneo) y demostración del **Visualizador 1** (MIP 3D). | **1 min 30 s** (2:50 – 4:20) |

---

# 🎙️ Guion de Sustentación

---

### PARTE 1: Integrante 1 — Yesenia Díaz
**Diapositivas asignadas:** 1, 2 y 3  
**Tiempo cronometrado:** 0:00 – 1:25 (1 min 25 s)

#### [0:00 – 0:25] Diapositiva 1: Portada y Contexto
*(Visual: Diapositiva 1 - Proyecto 2: Pengwin)*
> "Buenos días profesor y compañeros. Nuestro equipo está compuesto por Juan Pablo Maya, César Reyes y mi persona, Yesenia Díaz. Presentamos la entrega de la Semana 1 correspondiente a la Semana 8 del curso en el Proyecto Integrador PENGWIN.
>
> Este reto aborda la detección, segmentación y medición métrica de fracturas pélvicas en tomografía computarizada. Recordamos nuestra advertencia regulatoria obligatoria: este proyecto tiene un propósito estrictamente académico, no es un dispositivo médico y no está validado clínicamente para decisiones quirúrgicas."

#### [0:25 – 0:55] Diapositiva 2: Interpretación DATASET / Separación del Fragmento (GT)
*(Visual: Diapositiva 2 - Tablas resumen)*
> "Curamos 100 volúmenes de CT bajo la taxonomía anatómica oficial: Sacro en línea media, Coxal Izquierdo y Coxal Derecho. En el pipeline de carga estandarizamos la orientación canónica a LPS con SimpleITK para corregir la inversión de lateralidad entre pacientes.
>
> En la tabla superior vemos que los coxales presentan la mayor tasa de fractura, superando el 64% de los casos.
>
> En la tabla inferior analizamos los 275 fragmentos conminutos. Un hallazgo clínico fundamental es que el **88.7% de los fragmentos están en contacto directo con su fragmento principal**.
>
> Para cuantificar su separación implementamos la transformada de distancia Euclidiana `scipy.ndimage.distance_transform_edt` usando el espaciado físico real del header en milímetros por vóxel, jamás píxeles."

#### [0:55 – 1:25] Diapositiva 3: Gráfica 1 (EDA de Fragmentos y Volúmenes)
*(Visual: Diapositiva 3 - Los 3 subplots de distribución)*
> "En la Diapositiva 3 profundizamos en la complejidad morfológica con la Gráfica 1.
>
> A la izquierda observamos la distribución de fragmentos: el coxal izquierdo llega a registrar hasta 6 fragmentos individuales en un solo caso.
>
> En el panel central evidenciamos que la gran mayoría de pacientes sufren fracturas en dos o tres huesos simultáneamente, reflejando traumatismos de alta energía.
>
> Y en el panel derecho analizamos los volúmenes en escala logarítmica: la presencia de microfragmentos menores a 1 mL representa un reto crítico de segmentación para evitar que se desvanezcan en la red convolucional. Le doy paso a mi compañero Juan Pablo."

---

### PARTE 2: Integrante 2 — Juan Pablo Maya
**Diapositivas asignadas:** 4, 5 y 6  
**Tiempo cronometrado:** 1:25 – 2:50 (1 min 25 s)

#### [1:25 – 1:55] Diapositiva 4: Cortes Axiales y Desbalance 2D
*(Visual: Diapositiva 4 - Tabla de cortes axiales)*
> "Gracias, Yesenia. Dado que nuestra arquitectura operará corte por corte en 2D, fue indispensable estudiar la distribución espacial axial.
>
> De un total de 32,106 cortes en los 100 pacientes, encontramos que **un 24.3% de los cortes no contienen ningún hueso pélvico**. Asimismo, el sacro se encuentra concentrado en solo 16,052 cortes respecto a los más de 23,000 de los coxales.
>
> Conocer este desbalance es vital para la Semana 9: nos previene sobre el riesgo de colapso de la cabeza de detección ante cortes de fondo vacío y orienta la calibración del muestreo y la función de pérdida."

#### [1:55 – 2:25] Diapositiva 5: Gráfica 2 (Separación de Fragmentos)
*(Visual: Diapositiva 5 - Histograma y Scatter)*
> "En la Gráfica 2 analizamos la separación métrica de los fragmentos desplazados que no están en contacto.
>
> En el histograma de la izquierda vemos que la mayoría se desplazan entre 2 y 15 mm, pero en los coxales existen colas severas que alcanzan los 50 y hasta 73 mm de diástasis.
>
> En el gráfico de dispersión de la derecha correlacionamos el volumen frente a la distancia: los fragmentos más pequeños son precisamente los que experimentan mayores desplazamientos fuera de su lecho anatómico. Este análisis establece el estándar ground truth contra el cual mediremos el error de predicción en la Semana 10."

#### [2:25 – 2:50] Diapositiva 6: Intensidad del Hueso Anotado y Ventaneo HU
*(Visual: Diapositiva 6 - Tabla de percentiles HU)*
> "En la Diapositiva 6 evaluamos la distribución en Unidades Hounsfield dentro de las máscaras de referencia.
>
> Descubrimos que **en promedio el 36.8% del volumen óseo anotado se encuentra por debajo del umbral clásico de 200 HU**, particularmente por el hueso esponjoso trabecular del sacro y en las zonas fisuradas.
>
> Si aplicáramos un umbral binario duro perderíamos más de un tercio del hueso. Por ello, justificamos alimentar a la red con una ventana ósea normalizada a [0, 1] entre [-500 y 1300] HU, correspondiente a un nivel de 400 y ancho de 1800. Continúa mi compañero César."

---

### PARTE 3: Integrante 3 — César Reyes
**Diapositivas asignadas:** 7, 8 y 9  
**Tiempo cronometrado:** 2:50 – 4:20 (1 min 30 s)

#### [2:50 – 3:20] Diapositiva 7: División Dataset (Splits Fijos)
*(Visual: Diapositiva 7 - Tabla de splits train/val/test)*
> "Gracias, Juan Pablo. Para garantizar rigor experimental y prevenir fuga de datos, realizamos la partición **estrictamente por paciente completo**, nunca asignando cortes del mismo paciente a conjuntos distintos.
>
> Fijamos la semilla en 42 y estratificamos según la cantidad de huesos fracturados en proporción 70% entrenamiento, 15% validación y 15% prueba. Como se aprecia en la tabla, la densidad de fragmentos por caso se mantiene balanceada en torno a 5.7 y 6.0 en los tres conjuntos, asegurando que validación y test contengan fracturas complejas. Este split quedó congelado en `splits.json` para garantizar total reproducibilidad."

#### [3:20 – 3:50] Diapositiva 8: Caso 001 (Comparación de Ventaneo)
*(Visual: Diapositiva 8 - Los 4 paneles del corte axial z=134)*
> "En la Diapositiva 8 ilustramos el preprocesamiento en el corte axial 134 del Caso 001.
>
> El primer panel muestra la imagen en HU crudo.
>
> El segundo panel aplica ventana de tejido blando, saturando el hueso y perdiendo todo detalle interno.
>
> El tercer panel muestra nuestra ventana ósea calibrada: observen cómo se diferencia nítidamente la cortical densa, el trabecular interno y la línea de fractura sin artefactos.
>
> El cuarto panel sobrepone las máscaras ground truth por región anatómica. Esta entrada normalizada a float32 de 0 a 1 es el tensor exacto que ingresará a la arquitectura."

#### [3:50 – 4:20] Diapositiva 9: Visualizador 1 (MIP Rotatorio) y Cierre
*(Visual: Diapositiva 9 - Reproducir el video/animación del MIP)*
> "Finalmente, presentamos el Visualizador 1: reconstrucción 3D del volumen crudo mediante Proyección de Máxima Intensidad (MIP).
>
> Se trata de preprocesamiento clásico sin intervención de modelos. Remuestreamos el volumen a un vóxel isotrópico de 2 mm para evitar deformaciones longitudinales, aplicamos umbral óseo y proyectamos en 360 grados.
>
> Como se aprecia en la animación interactiva desarrollada en Plotly, el ortopedista puede rotar el volumen mediante el slider o reproducirlo de forma fluida, inspeccionando la arquitectura espacial del traumatismo antes de cualquier corte.
>
> Cumplimos así el 100% de los requisitos de la Semana 1 y quedamos preparados para la Semana 9 con el backbone FundidoraPC, atención CBAM y cabeza de detección con NMS propio. Muchas gracias."

---

## 💡 Consejos Clave para la Grabación del Video

1. **Mantener el ritmo constante:** Hablen con seguridad y calma (~120 palabras por minuto). El texto ya está medido para durar exactamente **4:20**, lo que les da 40 segundos de tranquilidad antes de los 5:00 minutos.
2. **Video de la Diapositiva 9:** Al pasar a la diapositiva 9, asegúrense de que el reproductor de PowerPoint inicie el clip (`media1.mp4`) o inicien la reproducción inmediatamente con un clic.
3. **Copia de seguridad:** Como lo solicita el profesor, guarden el video en formato MP4 (1080p) tanto en Google Drive como en una memoria local antes de la clase.
