# Demostración local y enlace temporal

**Estado de esta intervención:** no se inició Cloudflare porque el permiso de ejecución/conexión fue rechazado. La autorización del usuario para publicar está registrada, pero no sustituye ese permiso del entorno. No hay URL pública comprobada. La copia de trabajo sigue accesible localmente en `http://127.0.0.1:8766` durante esta sesión. Consultar `ESTADO_ENTREGA.md`.

El usuario autorizó explícitamente el enlace temporal de Cloudflare. Se publica únicamente el contenido de `dashboard/`: visualizaciones de seis casos del dataset, métricas, informe y video. El servidor no tiene acceso web al resto del proyecto ni ofrece inferencia sobre archivos subidos.

## Abrir solo en este equipo

Desde PowerShell en la carpeta del proyecto:

```powershell
$cfgDemo = Get-Content config_datos.local.json -Raw | ConvertFrom-Json
& $cfgDemo.python scripts/servir_dashboard.py --port 8765
```

Abrir `http://127.0.0.1:8765`. Para ese servidor de primer plano, Ctrl+C lo detiene.

## Compartir temporalmente

`scripts/iniciar_demo.ps1` inicia en segundo plano el servidor local y la herramienta oficial `tools/cloudflared.exe`. El puerto debe estar libre. No se instala un servicio permanente ni se configura inicio automático. La URL pública aparecerá en `salidas/cierre/demo_runtime/cloudflare_error.log`; la comprobación realizada se registra por separado en `salidas/cierre/despliegue.json`.

```powershell
./scripts/iniciar_demo.ps1
```

Para detener únicamente los procesos registrados de esta demostración:

```powershell
./scripts/detener_demo.ps1
```

El script comprueba la identidad de los procesos antes de detenerlos. Conserva el registro; para iniciar otra demo después de detenerla, archivar `salidas/cierre/demo_runtime` con otro nombre. La URL cambia en una nueva ejecución. No publicar una URL como disponible si no se ha comprobado.

El enlace es accesible para cualquiera que lo conozca y funciona mientras los procesos y la conexión de este equipo sigan activos. Es una demostración temporal, no alojamiento permanente. [Documentación oficial de Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

Herramienta descargada desde el enlace de Windows de la [documentación oficial](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/): cloudflared 2026.10.0. SHA256: `86aee4017b26625cee8484c113558f48effa4cd47f7aa05fcf425604e5d2b23c`.
