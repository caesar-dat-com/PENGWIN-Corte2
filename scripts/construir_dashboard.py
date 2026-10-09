"""Dashboard estático autocontenido: CT/MIP, cortes predichos y mallas físicas."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,SimpleITK as sitk
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from pengwin.io import resolver_volumen,normalizar_lps
from pengwin.viz_mip import figura_mip
from pengwin.dataset import extraer_bboxes_region

COLORS={1:'#ef714a',2:'#58a9ff',3:'#36c7aa'}
NAMES={1:'Sacro',2:'Coxal izquierdo',3:'Coxal derecho'}
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def write_plot(fig,path):
    fig.update_layout(height=None,width=None)
    html=fig.to_html(include_plotlyjs='../../assets/plotly.min.js',config={'responsive':True,'displaylogo':False})
    html=html.replace('<head>','<head><style>html,body{margin:0;height:100%;background:#111d2c;overflow:hidden}body>div{height:100%}</style>',1)
    path.write_text(html,encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',required=True);p.add_argument('--results-root',default='salidas/cierre/volumenes');p.add_argument('--output',default='dashboard');a=p.parse_args()
    dest=ROOT/a.output;(dest/'assets').mkdir(parents=True,exist_ok=True)
    (dest/'assets/plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    cfg=json.loads((ROOT/'config_datos.local.json').read_text(encoding='utf-8'));catalog=[]
    for cid in a.cases:
        src=ROOT/a.results_root/cid;out=dest/'data'/cid;(out/'cortes').mkdir(parents=True,exist_ok=True)
        ref=sitk.ReadImage(str(src/'instancias.mha'));inst=sitk.GetArrayFromImage(ref)
        gray=sitk.GetArrayFromImage(sitk.ReadImage(str(src/'ct_ventana.mha')))
        reference_ids=sitk.GetArrayFromImage(sitk.ReadImage(str(src/'gt_evaluacion.mha')))
        mapping={int(k):v for k,v in json.loads((src/'instancias.json').read_text(encoding='utf-8')).items()}
        ds=json.loads((src/'distancias.json').read_text(encoding='utf-8'));dist={d['instance']:d for d in ds}
        boxes=json.loads((src/'boxes.json').read_text(encoding='utf-8'));summary=json.loads((src/'resumen.json').read_text(encoding='utf-8'));meta=json.loads((src/'prediccion.json').read_text(encoding='utf-8'))
        slices=[]
        for z in range(len(gray)):
            rgb=np.repeat(gray[z,...,None],3,axis=2).astype(float);annotations=[]
            for label in np.unique(inst[z]):
                if not label:continue
                region=mapping[int(label)];color=np.array([int(COLORS[region][i:i+2],16) for i in (1,3,5)])
                mask=inst[z]==label;rgb[mask]=.55*rgb[mask]+.45*color
                center=ndi.center_of_mass(mask);d=dist.get(int(label))
                annotations.append({'id':int(label),'region':region,'x':float(center[1]),'y':float(center[0]),'label':f'F{label} · '+(f"{d['distance_surface_voxel_centers_mm']:.2f} mm" if d else 'principal'),'contact_26':d['contact_26'] if d else False})
            Image.fromarray(np.clip(rgb,0,255).astype(np.uint8)).save(out/'cortes'/f'{z:04d}.png')
            contour=np.zeros(gray[z].shape,bool)
            for region in (1,2,3):
                mask=(reference_ids[z]>=(region-1)*10+1)&(reference_ids[z]<=region*10)
                contour|=mask&~ndi.binary_erosion(mask)
            rgba=np.zeros((*contour.shape,4),np.uint8);rgba[contour]=255
            Image.fromarray(rgba).save(out/'cortes'/f'{z:04d}_referencia.png')
            reference_boxes=extraer_bboxes_region(reference_ids[z])
            slices.append({'z':z,'image':f'data/{cid}/cortes/{z:04d}.png','reference_image':f'data/{cid}/cortes/{z:04d}_referencia.png','annotations':annotations,'reference_boxes':[r['bbox'] for r in reference_boxes],'reference_labels':[r['clase_idx'] for r in reference_boxes],**boxes[z]})
        save(out/'cortes.json',slices)
        fig=go.Figure();orig=np.array(ref.GetOrigin());direction=np.array(ref.GetDirection()).reshape(3,3);sp=np.array(ref.GetSpacing())
        labels=[];xyz=[];labelcolors=[]
        objects=ndi.find_objects(inst)
        for label,region in mapping.items():
            sl=objects[label-1]
            if sl is None:continue
            crop=np.pad((inst[sl]==label).astype(np.float32),1)
            step=2 if np.count_nonzero(crop)>5000 else 1
            verts,faces,_,_=marching_cubes(crop,.5,spacing=tuple(sp[::-1]),step_size=step)
            offset=(np.array([s.start for s in sl])-1)*sp[::-1]
            points=(verts+offset)[:,::-1]@direction.T+orig
            d=dist.get(label);labeltext=f'F{label} · {NAMES[region]} · '+(f"{d['distance_surface_voxel_centers_mm']:.2f} mm" if d else 'principal')
            fig.add_trace(go.Mesh3d(x=points[:,0],y=points[:,1],z=points[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color=COLORS[region],opacity=.85,name=labeltext,hovertemplate=labeltext+'<extra></extra>',showlegend=False))
            center=np.array(ndi.center_of_mass(inst[sl]==label))+np.array([s.start for s in sl]);xyz.append((center[::-1]*sp)@direction.T+orig);labels.append(f'F{label} · '+(f"{d['distance_surface_voxel_centers_mm']:.2f} mm" if d else 'principal'));labelcolors.append(COLORS[region])
        if xyz:
            xyz=np.array(xyz);fig.add_trace(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],text=labels,mode='markers+text',textposition='top center',textfont={'size':11,'color':'#f1f5f9'},marker={'size':3,'color':labelcolors},name='Distancias',hoverinfo='text'))
        fig.update_layout(template='plotly_dark',paper_bgcolor='#111d2c',margin=dict(l=0,r=0,t=30,b=0),title=f'Caso {cid} · instancias predichas',scene=dict(aspectmode='data',xaxis_title='L (mm)',yaxis_title='P (mm)',zaxis_title='S (mm)'),showlegend=False)
        write_plot(fig,out/'malla.html')
        image=normalizar_lps(sitk.ReadImage(str(resolver_volumen(Path(cfg['images']),cid))));ct=sitk.GetArrayFromImage(image)
        mip=figura_mip(ct,image.GetSpacing()[::-1],cid,mm=3.,paso_grados=30)
        mip.update_layout(template='plotly_dark',paper_bgcolor='#111d2c',plot_bgcolor='#111d2c')
        write_plot(mip,out/'mip.html');del image,ct
        catalog.append({'id':cid,'detector':meta.get('detector','grid anterior'),'split':meta['split'],'slices':len(gray),'summary':summary,'latency_ms':meta['latency_mean_ms'],'device':meta['device'],'spacing_xyz':meta['spacing_xyz'],'distances':ds})
        print('DASHBOARD',cid,len(gray),'cortes',flush=True)
    save(dest/'data/catalogo.json',catalog)
    report=ROOT/'doc/INFORME_CIERRE.md'
    if report.exists():(dest/'assets/informe.md').write_bytes(report.read_bytes())
    for file in ('index.html','app.js','style.css'):
        (dest/file).write_bytes((ROOT/'web'/file).read_bytes())
if __name__=='__main__':main()
