# Avance guardado — 8 de octubre de 2026

Versión activa: detector RPN propio con respaldo del grid, segmentador seleccionado previamente y separación por interfaces con umbral 0.7. Visor local: http://127.0.0.1:8766/ mientras esté activo el servidor.

Se corrigió la orientación vertical de la MIP, se añadieron marcadores anatómicos y comparación con el contorno de referencia. Se verificaron 906 cortes de los casos de validación 002, 012 y 028. No hay cajas duplicadas por región. Persisten 32 instancias sobrantes y 8 omitidas; las distancias no tienen pares válidos suficientes para acreditar precisión.

CLAHE y entrenamiento con negativos difíciles se probaron, pero no sustituyen el modelo del visor. CLAHE obtuvo mAP50 30.03% frente a 80.85%; después de adaptar el decoder tres épocas, Dice 56.77% frente a 68.46%. Encoder/detector/RPN no se reentrenaron en esa prueba. No se iniciarán más experimentos por instrucción del usuario. Pasaron 38 pruebas.

Esta copia de trabajo conserva el avance. La carpeta original de OneDrive fue actualizada y verificada. Consulte RECIBO_ACTUALIZACION.json; no se utilizó Git.

El ZIP de respaldo contiene código, documentación, configuraciones, resultados JSON, pesos de las variantes probadas y archivos del visor activo. No duplica tomografías, volúmenes MHA, cachés, videos ni ejecutables; esos archivos permanecen en sus ubicaciones actuales. ESTADO_AVANCE.json identifica la versión activa. El informe y el video antiguos del visor siguen identificados como anteriores.
