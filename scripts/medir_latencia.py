"""Latencia reproducible CPU/GPU. GPU ausente se reporta como no disponible."""
import sys,json,time,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from pengwin.models.sesion2 import PelvisSesion2
from entrenar_sesion2 import decode
from pengwin.metrics_avance3 import gate_semantic

def main():
    torch.set_num_threads(4);p=json.loads((ROOT/'salidas/cierre/protocolo_final.json').read_text(encoding='utf-8'));ck=torch.load(ROOT/p['checkpoint'],map_location='cpu',weights_only=False)
    cases=json.loads((ROOT/'splits/splits.json').read_text(encoding='utf-8'))['val'][:3];images=[]
    for cid in cases:
        with np.load(ROOT/f'salidas/revision_oct08/datos/{cid}.npz') as data:images.extend(data['images'])
    output={'torch':torch.__version__,'platform':platform.platform(),'threads':4,'batch':1,'resolution':[256,256],'scope':'entrada ya en dispositivo; forward, decode, NMS, gating y resultados en CPU; no IO ni reconstruccion3D','results':{}}
    for dev in ('cpu','cuda'):
        if dev=='cuda' and not torch.cuda.is_available():output['results'][dev]={'status':'unavailable','reason':'torch.cuda.is_available() es False; no es una latencia cero'};continue
        model=PelvisSesion2();model.load_state_dict(ck['model']);model.to(dev).eval();times=[]
        with torch.inference_mode():
            for _ in range(5):model(torch.zeros(1,3,256,256,device=dev))
            for im in images:
                x=torch.from_numpy(im)[None,None].repeat(1,3,1,1).to(dev)
                if dev=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();out=model(x);b,s,c=[v.cpu().numpy() for v in decode(out)];gate_semantic(out['mascaras'][0].softmax(0).cpu().numpy(),b,c)
                if dev=='cuda':torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
        output['results'][dev]={'status':'measured','n':len(times),'mean_ms':float(np.mean(times)),'p50_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'samples_ms':times}
    (ROOT/'salidas/cierre/latencia.json').write_text(json.dumps(output,indent=2),encoding='utf-8');print(output['results'])
if __name__=='__main__':main()
