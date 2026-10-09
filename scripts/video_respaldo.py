"""Recorrido audiovisual de resultados reales (no simulación de inferencia en vivo)."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2,SimpleITK as sitk
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pengwin.io import resolver_volumen,normalizar_lps
from pengwin.viz_mip import a_isotropico,solo_hueso,mip_rotatorio

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='002');a=p.parse_args();cid=a.case
    src=ROOT/'salidas/cierre/volumenes'/cid;dash=ROOT/'dashboard/data'/cid;dest=ROOT/'salidas/cierre/demo_respaldo.webm'
    fontpath=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    fonts={s:ImageFont.truetype(str(fontpath),s) for s in (18,24,32,42)}
    colors={1:'#ef714a',2:'#58a9ff',3:'#36c7aa'};names={1:'Sacro',2:'Coxal izquierdo',3:'Coxal derecho'}
    writer=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'VP80'),12,(1280,720))
    if not writer.isOpened():raise RuntimeError('No hay codificador VP8')
    frames=0
    def frame(title,subtitle):
        im=Image.new('RGB',(1280,720),'#091321');d=ImageDraw.Draw(im);d.text((34,23),'PENGWIN · '+title,font=fonts[32],fill='white');d.text((34,67),subtitle,font=fonts[18],fill='#b5c9dc');d.text((34,685),'Uso académico. Sin validación clínica. Resultados guardados; no inferencia en vivo.',font=fonts[18],fill='#e2bf83');return im
    def emit(im,seconds):
        nonlocal frames
        arr=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
        for _ in range(round(seconds*12)):writer.write(arr);frames+=1
    im=frame('Demostración de respaldo',f'Caso {cid} · CT real · Orientación LPS')
    d=ImageDraw.Draw(im)
    for y,t in zip((210,300,390),('1. Volumen original: MIP del hueso','2. Cortes: cajas, máscaras y distancias','3. Reconstrucción de fragmentos en 3D')):d.text((70,y),t,font=fonts[32],fill='white')
    emit(im,3)
    cfg=json.loads((ROOT/'config_datos.local.json').read_text(encoding='utf-8'));ctimage=normalizar_lps(sitk.ReadImage(str(resolver_volumen(Path(cfg['images']),cid))))
    ct=sitk.GetArrayFromImage(ctimage);volume=solo_hueso(a_isotropico(ct,ctimage.GetSpacing()[::-1],3.),200);del ct,ctimage
    for angle,mip in zip(range(0,360,30),mip_rotatorio(volume,range(0,360,30))):
        im=frame('1 / Volumen original',f'Hueso ≥ 200 HU · sin modelo · rotación {angle}°')
        pixels=np.rint(np.clip((mip-200)/1300,0,1)*255).astype(np.uint8);picture=Image.fromarray(pixels).convert('RGB');scale=min(1150/picture.width,550/picture.height);picture=picture.resize((round(picture.width*scale),round(picture.height*scale)),Image.Resampling.LANCZOS);im.paste(picture,((1280-picture.width)//2,105));emit(im,.5)
    rows=json.loads((dash/'cortes.json').read_text(encoding='utf-8'))
    display_spacing=json.loads((src/'prediccion.json').read_text(encoding='utf-8'))['spacing_xyz'];ratio=display_spacing[1]/display_spacing[0]
    pw=min(540,round(540/ratio));ph=round(pw*ratio);px=45+(540-pw)//2;py=115+(540-ph)//2
    for z in np.unique(np.rint(np.linspace(0,len(rows)-1,60)).astype(int)):
        row=rows[z];im=frame('2 / Inferencia corte a corte',f'Corte {z+1}/{len(rows)} · etiquetas en mm respecto al principal de la región')
        picture=Image.open(ROOT/'dashboard'/row['image']).convert('RGB').resize((pw,ph));im.paste(picture,(px,py));d=ImageDraw.Draw(im)
        for box,r in zip(row['boxes'],row['labels']):d.rectangle((px+box[0]*pw,py+box[1]*ph,px+box[2]*pw,py+box[3]*ph),outline=colors[r+1],width=2)
        for i,ann in enumerate(row['annotations']):
            x=min(420,px+ann['x']*pw/256);y=min(625,py+ann['y']*ph/256);text=ann['label'];bounds=d.textbbox((x,y),text,font=fonts[18]);d.rectangle(bounds,fill='#091321');d.text((x,y),text,font=fonts[18],fill=colors[ann['region']])
            if i<12:d.text((640,130+i*39),text+' · '+names[ann['region']],font=fonts[18],fill=colors[ann['region']])
        if not row['annotations']:d.text((640,160),'Sin instancias predichas en este corte.',font=fonts[24],fill='white')
        emit(im,.25)
    ref=sitk.ReadImage(str(src/'instancias.mha'));inst=sitk.GetArrayFromImage(ref);mapping={int(k):v for k,v in json.loads((src/'instancias.json').read_text(encoding='utf-8')).items()};sp=np.array(ref.GetSpacing());origin=np.array(ref.GetOrigin());direction=np.array(ref.GetDirection()).reshape(3,3)
    distances={r['instance']:r for r in json.loads((src/'distancias.json').read_text(encoding='utf-8'))};meshes=[];allpoints=[]
    for label,region in mapping.items():
        coordinates=np.argwhere(inst==label)
        if not len(coordinates):continue
        lo=coordinates.min(0);hi=coordinates.max(0)+1;sl=tuple(slice(a,b) for a,b in zip(lo,hi));crop=np.pad((inst[sl]==label).astype(np.float32),1)
        verts,faces,_,_=marching_cubes(crop,.5,spacing=tuple(sp[::-1]),step_size=3 if len(coordinates)>5000 else 1)
        points=(verts+(lo-1)*sp[::-1])[:,::-1]@direction.T+origin;center=(coordinates.mean(0)[::-1]*sp)@direction.T+origin;allpoints.append(points);meshes.append((points,faces,label,region,center))
    allpoints=np.concatenate(allpoints) if allpoints else np.array([[0,0,0],[1,1,1]])
    lo=allpoints.min(0);hi=allpoints.max(0);span=np.maximum(hi-lo,1)
    for angle in range(0,360,30):
        fig=plt.figure(figsize=(12.8,7.2),dpi=100,facecolor='#091321');ax=fig.add_subplot(111,projection='3d',facecolor='#091321')
        for points,faces,label,region,center in meshes:
            ax.add_collection3d(Poly3DCollection(points[faces],facecolors=colors[region],edgecolors='none',alpha=.88))
            distance=distances.get(label);text=f'F{label} '+(f"{distance['distance_surface_voxel_centers_mm']:.1f} mm" if distance else 'principal')
            ax.text(*center,text,color='white',fontsize=7)
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_box_aspect(span);ax.view_init(elev=20,azim=angle);ax.set_axis_off()
        fig.text(.026,.95,'PENGWIN · 3 / Fragmentos predichos en 3D',fontsize=20,color='white');fig.text(.026,.91,'Malla simplificada para visualizar; las distancias se calculan en el volumen.',fontsize=12,color='#b5c9dc');fig.text(.026,.025,'Uso académico. Sin validación clínica. Las predicciones pueden omitir o fusionar fragmentos.',fontsize=11,color='#e2bf83')
        fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());plt.close(fig);emit(im,.5)
    summary=json.loads((src/'resumen.json').read_text(encoding='utf-8'));im=frame('Resultados y límites',f'Caso {cid} · evaluación contra máscara de referencia');d=ImageDraw.Draw(im)
    lines=[f"Referencia: {summary['gt']} fragmentos. Predicción: {summary['pred']} instancias.",f"Omisiones: {summary['missed']}. Instancias sobrantes: {summary['extra']}.",f"Pares válidos para distancia: {summary['distance_valid_pairs']}.",'La máscara macro y la separación individual se evalúan por separado.']
    for y,t in zip((200,290,380,470),lines):d.text((50,y),t,font=fonts[24],fill='white')
    emit(im,3);writer.release()
    reader=cv2.VideoCapture(str(dest));actual=int(reader.get(cv2.CAP_PROP_FRAME_COUNT));ok,first=reader.read();reader.release()
    if not ok or actual!=frames:raise RuntimeError('Video no verificable')
    (ROOT/'salidas/cierre/video.json').write_text(json.dumps({'file':str(dest.relative_to(ROOT)),'frames':frames,'fps':12,'seconds':frames/12,'codec':'VP8','scope':'Recorrido renderizado de resultados reales, sin audio; no grabacion de una inferencia en vivo'},indent=2),encoding='utf-8')
    print('VIDEO',frames/12,'segundos',flush=True)
if __name__=='__main__':main()
