"""Secuencia de esta ejecución local; no tarea programada ni automatización recurrente."""
import sys,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'salidas/cierre'
def wait_for(path):
    start=time.monotonic()
    while not path.exists():
        if time.monotonic()-start>7200:raise TimeoutError(str(path))
        time.sleep(5)
def run(script,*args):
    print('FASE',script,*args,flush=True)
    (OUT/'estado_ejecucion.json').write_text(json.dumps({'phase':script,'args':args,'status':'running'}),encoding='utf-8')
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True)
def main():
    wait_for(OUT/'ajuste/comparacion.json')
    run('ajustar_cajas.py')
    splits=json.loads((ROOT/'splits/splits.json').read_text(encoding='utf-8'))
    run('predecir_volumen_cierre.py','--cases',*splits['val'][:3])
    run('evaluar_instancias_cierre.py')
    run('predecir_volumen_cierre.py','--cases',*splits['test'][:3],'--include-test')
    run('evaluar_instancias_cierre.py','--test')
    run('comparar_sam_cierre.py','--include-test')
    wait_for(OUT/'ablaciones/comparacion.json')
    run('medir_latencia.py')
    (OUT/'estado_ejecucion.json').write_text(json.dumps({'phase':'experimentos','status':'completed'}),encoding='utf-8')
if __name__=='__main__':main()
