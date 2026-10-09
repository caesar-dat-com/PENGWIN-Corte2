"""Construye evidencias al terminar los experimentos de esta ejecución local."""
import sys,json,time,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    start=time.monotonic()
    while True:
        p=ROOT/'salidas/cierre/estado_ejecucion.json'
        if p.exists() and json.loads(p.read_text(encoding='utf-8')).get('status')=='completed':break
        if time.monotonic()-start>7200:raise TimeoutError('Experimentos incompletos')
        time.sleep(5)
    for script,args in [('documentar_cierre.py',[]),('construir_dashboard.py',['--cases','002','012','028','004','006','010']),('video_respaldo.py',['--case','002'])]:
        print('ENTREGABLE',script,flush=True)
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True)
    shutil.copy2(ROOT/'salidas/cierre/demo_respaldo.webm',ROOT/'dashboard/assets/demo_respaldo.webm')
    (ROOT/'salidas/cierre/estado_entregables.json').write_text(json.dumps({'status':'completed'}),encoding='utf-8')
