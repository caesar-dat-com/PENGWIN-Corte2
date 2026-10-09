"""Servidor solo local, limitado a dashboard; no expone dataset ni pesos."""
import argparse
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    folder=ROOT/'dashboard'
    if not (folder/'index.html').exists():raise SystemExit('Generar el dashboard antes de iniciar')
    server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(SimpleHTTPRequestHandler,directory=str(folder)))
    print(f'Dashboard local: http://127.0.0.1:{a.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:server.server_close()
