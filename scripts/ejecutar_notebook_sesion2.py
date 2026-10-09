"""Ejecuta únicamente el notebook de inspección local; captura salidas reales."""
from pathlib import Path
import os,io,contextlib,base64
import nbformat as n
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    path=ROOT/'notebooks/Semana3_Sesion2_Fragmentos.ipynb'
    nb=n.read(path,as_version=4);scope={};count=0
    os.chdir(ROOT)
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        count+=1;outputs=[];buf=io.StringIO()
        def capture(*args,**kwargs):
            for num in plt.get_fignums():
                image=io.BytesIO();plt.figure(num).savefig(image,format='png',bbox_inches='tight')
                outputs.append(n.v4.new_output('display_data',data={'image/png':base64.b64encode(image.getvalue()).decode('ascii')}))
            plt.close('all')
        plt.show=capture
        with contextlib.redirect_stdout(buf):exec(compile(cell.source,str(path),'exec'),scope)
        text=buf.getvalue()
        if text:outputs.insert(0,n.v4.new_output('stream',name='stdout',text=text))
        cell.outputs=outputs;cell.execution_count=count
    n.validate(nb);n.write(nb,path)
    print('Notebook de inspección ejecutado:',count,'celdas, sin errores')
if __name__=='__main__':main()
