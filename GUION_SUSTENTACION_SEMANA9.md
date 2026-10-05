# Guion de Sustentación — Semana 9 (Avance 2 del Proyecto)
## Proyecto Integrador: Sistema PENGWIN
**Detección Multiobjetivo y Segmentación de Fracturas Pélvicas en CT**  
**Analítica de Datos · UAO 2026-2 · Prof. Carlos A. Ferro**  
**Equipo:** Yesenia Díaz · Juan Pablo Maya · César Reyes  
**Límite de tiempo:** Máximo 7:00 minutos | **Tiempo objetivo calibrado:** 5:30 minutos (margen de 1:30 min)

---

## 📋 Distribución de Roles, Tiempos y Visuales

| Integrante | Eje Temático | Visual / Imagen a Mostrar | Código a Respaldar | Tiempo Asignado |
|---|---|---|---|:---:|
| **1. Yesenia Díaz** | Contexto PENGWIN, reglas del curso, ventaneo HU clínico en `float32` y justificación anatómica del filtro de ruido $N < 15$. | Carátula del cuaderno y figura `salidas/inventario_ruido_componentes.png` | `pengwin/io.py`<br>`pengwin/dataset.py` | **1 min 45 s**<br>(0:00 – 1:45) |
| **2. Juan Pablo Maya** | Arquitectura: Backbone `FundidoraPC` con doble convolución, CBAM con kernel $9 \times 9$, cabeza en Grid de 8 canales y NMS propio sin cajas negras. | Sección 2, 3, 4 y 5 del cuaderno Jupyter | `pengwin/models/backbone.py`<br>`pengwin/models/cbam.py`<br>`pengwin/models/detector.py`<br>`pengwin/models/nms.py` | **1 min 55 s**<br>(1:45 – 3:40) |
| **3. César Reyes** | Función de pérdida compuesta con Focal Loss ($\gamma = 0.5$), prueba de overfit sobre Caso 001 real, visualización anatómica con alto contraste y hoja de ruta Semana 10. | Gráfica de pérdida `salidas/overfit_loss_curve.png` y predicciones reales `salidas/overfit_predicciones_boxes.png` | `pengwin/models/loss.py`<br>`scripts/run_overfit_demo.py` | **1 min 50 s**<br>(3:40 – 5:30) |

---

## 🧠 Base Semi-Teórica Fundamental (Preguntas Típicas del Docente)

1. **¿Por qué ventana HU $[-500, +1300]$ y formato `float32`?**
   * *Teoría:* Las Unidades Hounsfield (HU) miden la atenuación radiológica relativa al agua ($0\text{ HU}$) y al aire ($-1000\text{ HU}$). El tejido blando se sitúa entre $+40$ y $+80\text{ HU}$, y el hueso entre $+200$ y $+1500\text{ HU}$.
   * *Justificación:* Al aplicar nivel $L=400$ y ancho $W=1800$, la ventana resultante es $[-500, +1300]\text{ HU}$. Se normaliza a $[0, 1]$ para que la red trabaje con gradientes estables. Se procesa en `float32` para evitar el desbordamiento numérico que ocurre en `int16` cuando hay implantes o vóxeles $>32767\text{ HU}$ (demostrado en el Caso 080).

2. **¿Por qué es seguro descartar componentes con $N < 15$ vóxeles?**
   * *Teoría:* En cortes 2D, el efecto de volumen parcial o pequeños artefactos metálicos generan componentes espurios aislados.
   * *Evidencia:* La auditoría volumétrica 3D demostró que el fragmento conminuto real más diminuto de todo el dataset tiene **468 vóxeles** ($0.313\text{ mL}$, Caso 85). Quince vóxeles equivalen a solo $\sim 0.01\text{ mL}$ ($30$ veces menor que el fragmento real más pequeño). Filtrar $N < 15$ descarta 1,506 instancias de ruido espurio ($2.4\%$) y conserva intactas 61,232 instancias reales ($97.6\%$).

3. **¿Por qué filtro de $9 \times 9$ en la atención espacial (CBAM)?**
   * *Teoría:* El módulo de atención espacial (SAM) comprime canales con AvgPool y MaxPool y aplica una convolución con sigmoide para determinar *dónde enfocar*.
   * *Mejora:* Ampliar el kernel de $7 \times 7$ a $9 \times 9$ incrementa el campo receptivo en un $+65.3\%$ ($81$ vs $49$ celdas). Huesos como el ilion y el sacro son estructuras continuas y alargadas; un filtro más amplio permite abarcar la curvatura anatómica completa y correlacionar fragmentos desplazados.

4. **¿Por qué cuadrícula (Grid) y no cajas de anclaje (*Anchors*)?**
   * *Teoría:* Las cajas de anclaje fijas (como en Faster R-CNN o YOLOv3) introducen hiperparámetros artificiales de escala y relación de aspecto.
   * *Diseño:* Nuestra cabeza convolucional divide el espacio latente en una cuadrícula de $16 \times 16$ celdas donde cada celda predice directamente si contiene un centro óseo, las coordenadas normalizadas con sigmoide y los logits de clase. Emite exactamente **8 canales** ($1$ de objetidad, $4$ de coordenadas y $3$ de clase), cumpliendo la restricción de **no superar 10 canales en la reconstrucción latente**.

5. **¿Qué es y para qué sirve "Gama 0.5" ($\gamma = 0.5$)?**
   * *Teoría:* Proviene de la **Focal Loss** ($\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$).
   * *Problema:* En la tomografía, más del $95\%$ de las celdas del grid son fondo vacío. Con entropía cruzada común, el gradiente acumulado del fondo ahoga la señal de los huesos.
   * *Solución:* Con $\gamma = 0.5$, el factor modulador $(1 - p_t)^{0.5}$ atenúa el error en las celdas de fondo fáciles y concentra la señal en las fronteras óseas y fracturas complejas.

6. **¿Por qué la prueba de Overfit intencional es el entregable de la Semana 9?**
   * *Metodología:* La rúbrica oficial exige demostrar que la red es un aproximador universal válido capaz de memorizar y converger a cero sobre un lote pequeño de cortes reales representativos. El entrenamiento masivo sobre los 100 pacientes corresponde a la Semana 10 (con la segmentación de instancias).

---

# 🎙️ Guion de Sustentación por Participante

---

### PARTE 1: Integrante 1 — Yesenia Díaz
* **Tiempo asignado:** `0:00 – 1:45` (1 min 45 s)
* **Visual en pantalla:**
  1. Carátula del cuaderno: `notebooks/Semana2_Backbone_CBAM_Deteccion.ipynb`.
  2. Infografía de ruido: `salidas/inventario_ruido_componentes.png`.

#### [0:00 – 0:45] Portada, Contexto y Reglas de Juego
> "Buenos días profesor y compañeros. Junto a Juan Pablo Maya y César Reyes, presentamos el Avance 2 correspondiente a la Semana 9 del proyecto integrador PENGWIN.
>
> Nuestro objetivo en esta entrega es implementar la detección multiobjetivo de las tres regiones pélvicas principales: Sacro, Coxal Izquierdo y Coxal Derecho mediante cajas delimitadoras (*bounding boxes*).
>
> Cumpliendo estrictamente las normas del curso, **no empleamos librerías ni modelos de caja negra comerciales** como YOLO o Detectron2. Toda la arquitectura —backbone, mecanismo de atención, cabeza de detección en cuadrícula, algoritmo NMS y función de pérdida compuesta— fue implementada y validada por nuestro equipo desde cero."

#### [0:45 – 1:45] Ventaneo Óseo Clínico y Filtro de Ruido ($N < 15$ vóxeles)
*(Cambiar a la imagen `salidas/inventario_ruido_componentes.png`)*
> "En el tratamiento de datos aplicamos dos correcciones rigurosas. Primero, calibramos la tomografía con la ventana ósea clínica de $[-500, +1300]\text{ HU}$ conservando los datos en `float32`. Esto garantiza que el aire sea negro, el tejido blando gris oscuro y el hueso blanco brillante, resolviendo de raíz el desbordamiento numérico que afectaba a casos con implantes densos.
>
> Segundo, atendiendo la recomendación sobre el ruido de segmentación, realizamos un análisis de sensibilidad sobre los 100 casos. Demostramos clínicamente que el fragmento conminuto real más diminuto en todo el dataset tiene **468 vóxeles** ($0.313\text{ mL}$, en el Caso 85).
>
> Por ello, adoptar el umbral de **ignorar componentes con menos de 15 vóxeles** es 100% seguro: eliminamos 1,506 manchas de ruido espurio por volumen parcial o metales (el $2.4\%$) y preservamos intactas el $97.6\%$ de las estructuras óseas reales. De esta manera, aseguramos que la red nunca intente aprender cajas falsas sobre ruido.
>
> Le doy la palabra a mi compañero Juan Pablo para explicar la arquitectura de la red."

---

### PARTE 2: Integrante 2 — Juan Pablo Maya
* **Tiempo asignado:** `1:45 – 3:40` (1 min 55 s)
* **Visual en pantalla:**
  1. Cuaderno Jupyter: Sección 2 (CBAM) y Sección 3 (Backbone `FundidoraPC`).
  2. Cuaderno Jupyter: Sección 4 (Grid Detection Head) y Sección 5 (NMS Propio).

#### [1:45 – 2:40] Backbone Residual y CBAM con Filtro $9 \times 9$
> "Continuando con la red neuronal, construimos el backbone propio `FundidoraPC` organizado en 4 etapas progresivas: de $3$ a $256$ canales. Siguiendo la recomendación del docente, cada etapa implementa una **doble convolución** con bloques residuales, comprimiendo la imagen espacialmente en un factor acumulado de 16 ($256 \to 16$).
>
> Justo antes de la bifurcación hacia las cabezas, conectamos el bloque de atención dual **CBAM**:
> - La atención de canal analiza qué canales de características de densidad ósea deben priorizarse.
> - Y para la atención espacial, aplicamos la recomendación del profesor de utilizar un **filtro convolucional amplio de $9 \times 9$** en lugar del $7 \times 7$ estándar.
>
> ¿Por qué $9 \times 9$? En tomografía de pelvis, huesos como el ilion son arcos extensos y curvados. Un kernel de $9 \times 9$ amplía el campo de visión espacial en un $65\%$, permitiendo a la atención abarcar la curvatura continua del hueso y correlacionar fragmentos fracturados que han sufrido desplazamiento."

#### [2:40 – 3:40] Cabeza de Detección en Grid y NMS Vectorial sin Librerías Externas
> "Para la localización espacial, diseñamos una cabeza con **cuadrícula propia de $16 \times 16$ celdas**.
>
> Cada celda predice un vector compacto de **8 canales**:
> - 1 canal de presencia ósea $P_{obj}$,
> - 4 canales para parámetros de caja $(t_x, t_y, t_w, t_h)$ decodificados con sigmoides respecto a la celda,
> - y 3 probabilidades para la clasificación anatómica.
> Con esto cumplimos estrictamente la directriz de **no sobrepasar 10 canales en la reconstrucción latente**, manteniendo la red ligera y rápida en memoria.
>
> Finalmente, dado que múltiples celdas vecinas pueden activarse para el mismo hueso, implementamos desde cero nuestro propio algoritmo de **Non-Maximum Suppression (NMS)** en PyTorch. Nuestro NMS calcula la matriz de IoU de forma vectorial y suprime redundancias de manera desacoplada clase por clase, garantizando que el sacro no inhiba al coxal adyacente.
>
> Doy paso a mi compañero César para presentar la función de pérdida y los resultados reales."

---

### PARTE 3: Integrante 3 — César Reyes
* **Tiempo asignado:** `3:40 – 5:30` (1 min 50 s)
* **Visual en pantalla:**
  1. Gráfica de convergencia: `salidas/overfit_loss_curve.png`.
  2. Figura de predicciones reales contrastadas: `salidas/overfit_predicciones_boxes.png`.

#### [3:40 – 4:35] Función de Pérdida con Gama 0.5 y Prueba de Overfit Real
> "Para entrenar la red formulamos una función de pérdida compuesta multitarea calibrada. En ella aplicamos el concepto de **Gama 0.5 ($\gamma = 0.5$)** dentro de la Focal Loss binaria.
>
> Como en una tomografía más del $95\%$ del volumen es fondo vacío o tejido blando sin hueso, una pérdida convencional se saturaría con el fondo fácil. Al fijar $\gamma = 0.5$, el factor modulador atenúa el gradiente de las celdas vacías obvias y multiplica el peso sobre las regiones óseas y bordes articulares difíciles.
>
> Para la prueba de correctitud exigida en la rúbrica de la Semana 9, ejecutamos un **overfit intencional sobre 4 cortes clínicos reales del Caso 001**, abarcando desde el nivel inferior hasta el rango medio-alto.
>
> Como vemos en la curva de pérdida, el modelo demostró estabilidad numérica total: la pérdida descendió de **$3.056$ a solo $0.055$** en 100 épocas, con una pérdida de regresión de cajas de cero."

#### [4:35 – 5:30] Resultados Visuales Clínicos Reales y Cierre
*(Cambiar a la imagen `salidas/overfit_predicciones_boxes.png`)*
> "En esta figura observamos los resultados con el contraste radiológico corregido, comparando el Ground Truth oficial en la fila superior contra las predicciones de nuestro modelo en la fila inferior:
>
> - En la **Muestra 0 ($z=134$, corte inferior de pubis y acetábulos)**: únicamente existen los coxales. El modelo detecta el Coxal Derecho con $0.79$ y el Izquierdo con $0.94$, **y predice cero cajas para el Sacro**, demostrando que no alucina estructuras inexistentes.
> - En las **Muestras 1, 2 y 3 ($z=220, 235$ y $250$, cortes medio a alto)**: el modelo localiza el Sacro en su posición anatómica real: **abajo, en la columna posterior** con scores de $0.95$ a $0.98$, y encuadra los coxales a los costados laterales sobre las alas ilíacas.
>
> Las líneas punteadas del modelo coinciden con exactitud milimétrica sobre las cajas del Ground Truth médico.
>
> Con esto certificamos la correctitud de la Semana 9. Para la Semana 10, completaremos la cabeza de segmentación de instancias para mapear hasta 15 clases de fragmentos, mediremos las distancias de fractura en milímetros con Transformada de Distancia Euclidiana y evaluaremos el baseline zero-shot con SAM.
>
> Muchas gracias."

---

### 💡 Checklist Final para la Grabación del Equipo:
1. **Pestañas abiertas antes de grabar:**
   * Cuaderno `notebooks/Semana2_Backbone_CBAM_Deteccion.ipynb`.
   * Visor de imagen con `salidas/inventario_ruido_componentes.png`.
   * Visor de imagen con `salidas/overfit_loss_curve.png` y `salidas/overfit_predicciones_boxes.png`.
2. **Puntero del mouse:** Apuntar a la fórmula de Focal Loss ($\gamma = 0.5$), al parámetro `kernel_size=9` en CBAM y a la posición posterior del Sacro en las muestras 1, 2 y 3.
