# Entrenamiento dirigido a falsos positivos anatómicos

Solicitud: continuar la mejora de identificación anatómica, conservando el visor como comparación.

Datos: 560 cortes reales de 70 pacientes train y 120 cortes de 15 pacientes val. No se generan imágenes sintéticas, no se usan casos test y no se eligen ejemplos de entrenamiento a partir de val. En train hay 467532 píxeles de fondo anotado con HU > 200, distribuidos en 559 cortes. Son candidatos a negativos difíciles; no se afirma que todos sean errores del modelo.

Se comparan control y pérdida adicional de negativos difíciles (peso 0.5), tres épocas cada uno, AdamW 1e-4, lote ocho, semilla 42, mismo punto de partida y orden de datos. Solo se entrena el decoder; encoder, detección y clasificación están congelados y se verifica igualdad exacta de sus parámetros y buffers. Se conserva el decoder de ocho canales.

Los candidatos son píxeles anotados como fondo con HU > 200 o probabilidad predicha de pelvis > 0.5. Se penalizan los de mayor entropía cruzada, hasta el 2% de los píxeles de cada corte. Los píxeles anotados como hueso objetivo nunca reciben esta penalización.

La evaluación utiliza las mismas cajas RPN con respaldo seleccionadas anteriormente. Criterios fijados antes de entrenar: Dice macro no inferior al anterior, caída máxima de Dice de 0.005 por región, caída máxima de sensibilidad al hueso de un punto porcentual y menor cantidad de píxeles de fondo falsamente marcados. Solo los candidatos elegibles pasan a comprobación volumétrica antes de actualizar el visor.

Se selecciona con val, por lo que los resultados no constituyen una evaluación independiente. Esta comparación no incorpora todavía cortes vecinos como entrada; esa sería otra modificación que requiere su propia validación. Resultados y estado: salidas/negativos_revision.

## Segunda intensidad
La primera comparación no produjo candidatos elegibles: el peso 0.5 redujo falsos positivos pero perdió sensibilidad y Dice del sacro. Se prueba peso 0.1, conservando tres épocas y todas las demás condiciones. Se reutiliza la comparación control ya ejecutada; no se repite ni se modifica test. Esta es una segunda selección iterativa en validación, no confirmación independiente.
