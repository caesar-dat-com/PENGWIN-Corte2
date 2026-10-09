# Actualización local autorizada — 8 de octubre de 2026

El usuario solicitó incorporar el avance a la carpeta original local. El instalador comprueba conflictos antes de escribir, respalda los archivos anteriores y verifica cada archivo copiado mediante SHA-256. El recibo cierre_instalado.json en el directorio de trabajo confirma el resultado; esta nota por sí sola no acredita instalación completada.

La versión activa y sus límites se describen en AVANCE_GUARDADO.md y ESTADO_AVANCE.json. Incluye RPN con respaldo, MIP corregida, 906 cortes de validación y separación revisada: 32 instancias sobrantes y 8 omitidas. CLAHE y negativos difíciles fueron evaluados y no desplegados. Se conservan los resultados históricos. No se modifica Git ni se publica el visor.

## Historial anterior a esta autorización

# Estado real de esta intervención

Las mejoras están preparadas y probadas en la copia de trabajo `.cierre-semana10`. **No se incorporaron a la carpeta original de OneDrive**: la solicitud de permiso para ejecutar el instalador fue rechazada antes de comenzar. No se creó respaldo de instalación porque esa acción no llegó a ejecutarse. No se utilizó Git.

El usuario autorizó un enlace temporal de Cloudflare, pero la solicitud posterior de permiso de ejecución/conexión también fue rechazada. **No hay una URL pública activa creada por esta intervención.** El explorador de la copia de trabajo se comprobó localmente en `http://127.0.0.1:8766`, disponible mientras el servidor de esta sesión permanezca activo.

## Trabajo terminado en la copia de trabajo

- Calibración del decoder: Dice anatómico val 60,45 % → 66,72 %; F1 de contornos 36,89 % → 52,89 %, sobre los mismos 120 cortes.
- Ajuste de cajas evaluado y descartado porque no preservó todas las métricas anatómicas exigidas por la regla de selección.
- Cuatro ablaciones controladas y evaluación SAM con protocolo congelado.
- Seis volúmenes completos: 1.763 cortes; tres val y tres test. No son 100 inferencias volumétricas.
- Dashboard con tres visualizadores y proporciones físicas, informe, fichas de modelos, guion y video de respaldo de 33 segundos.
- 25 pruebas funcionales aprobadas, más verificación de geometría, separación de pacientes, hashes, imágenes y video.
- Equipo de tres integrantes registrado como autorizado por el profesor, según confirmación del usuario.

## Límites pendientes

No se ha resuelto satisfactoriamente la separación individual: Dice por fragmento GT de 31,73–43,82 % en los tres volúmenes test, con omisiones y extras. No hubo pares correctos de fragmento y principal que permitan informar un MAE de distancia. La clasificación test sí alcanza F1 macro 91,23 % y AUC 97,98 %, pero no debe sustituir las métricas de segmentación. GPU no medida; Git y colaboración remota pendientes por instrucción del usuario.

El instalador preparado en el directorio de trabajo, `instalar_cierre.py`, verifica hashes, detecta conflictos y crea un ZIP de respaldo antes de copiar. El manifiesto es `cierre_manifest.json`. Aplicarlo requiere autorización para escribir en OneDrive. La publicación utiliza `scripts/iniciar_demo.ps1` y requiere autorización de conexión. No se intentó evitar las restricciones mediante otro mecanismo.
