'use strict';
const $=id=>document.getElementById(id),colors=['','#ef714a','#58a9ff','#36c7aa'],names=['','Sacro','Coxal izquierdo','Coxal derecho'];
let catalog=[],current=null,slices=[],generation=0,drawToken=0;
const pct=x=>x==null?'No evaluable':(100*x).toFixed(1)+' %';
async function loadCase(){
  const gen=++generation;++drawToken;slices=[];current=catalog.find(c=>c.id===$('case').value);
  const response=await fetch('data/'+current.id+'/cortes.json');if(!response.ok)throw Error('Cortes no disponibles');
  const rows=await response.json();if(gen!==generation)return;slices=rows;
  $('view').height=Math.round(768*current.spacing_xyz[1]/current.spacing_xyz[0]);
  $('slice').max=slices.length-1;$('slice').value=Math.floor(slices.length/2);
  $('scope').textContent=(current.split==='test'?'Prueba':'Validación')+' · '+current.slices+' cortes contiguos · orientación LPS · proporciones físicas';
  const s=current.summary;
  $('stats').innerHTML=[['Instancias predichas',s.pred],['Fragmentos de referencia',s.gt],['Dice de fragmentos',pct(s.dice_gt)],['Error de distancia',s.distance_MAE_mm==null?'No evaluable':s.distance_MAE_mm.toFixed(2)+' mm']].map(([label,value])=>'<div class="stat"><small>'+label+'</small><strong>'+value+'</strong></div>').join('');
  $('quality').textContent='En este volumen se detectaron '+s.extra+' instancias sobrantes y '+s.missed+' fragmentos omitidos. Hay '+s.distance_valid_pairs+' pares válidos para evaluar distancias. Latencia media del modelo y posprocesado 2D en '+current.device.toUpperCase()+': '+current.latency_ms.toFixed(1)+' ms por corte; no incluye lectura ni reconstrucción 3D.';
  $('mipframe').removeAttribute('src');$('meshframe').removeAttribute('src');showTab(document.querySelector('nav button.active').dataset.tab);draw();
}
function showTab(name){
  document.querySelectorAll('.panel').forEach(p=>p.hidden=p.id!==name);document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));
  if(current&&name==='mip'&&!$('mipframe').getAttribute('src'))$('mipframe').src='data/'+current.id+'/mip.html';
  if(current&&name==='mesh'&&!$('meshframe').getAttribute('src'))$('meshframe').src='data/'+current.id+'/malla.html';
}
function draw(){
  if(!slices.length)return;const token=++drawToken,row=slices[Number($('slice').value)];
  $('slice-label').textContent=(row.z+1)+' / '+slices.length;const image=new Image();
  image.onload=()=>{
    if(token!==drawToken)return;const canvas=$('view'),c=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
    c.clearRect(0,0,w,h);c.drawImage(image,0,0,w,h);c.lineWidth=2;
    c.font='bold 24px Segoe UI';c.textAlign='center';c.fillStyle='#ffffff';c.strokeStyle='#000000';c.lineWidth=4;
    [['A',w/2,28],['P',w/2,h-12],['R',16,h/2],['L',w-16,h/2]].forEach(([t,x,y])=>{c.strokeText(t,x,y);c.fillText(t,x,y);});c.textAlign='left';c.lineWidth=2;
    row.boxes.forEach((b,i)=>{c.strokeStyle=colors[row.labels[i]+1];c.strokeRect(b[0]*w,b[1]*h,(b[2]-b[0])*w,(b[3]-b[1])*h);});
    if($('reference').checked){c.setLineDash([8,5]);c.strokeStyle='#ffffff';c.lineWidth=2;(row.reference_boxes||[]).forEach(b=>c.strokeRect(b[0]*w,b[1]*h,(b[2]-b[0])*w,(b[3]-b[1])*h));c.setLineDash([]);}
    if($('anatomy').checked&&row.reference_image){const refImage=new Image();refImage.onload=()=>{if(token===drawToken&&$('anatomy').checked)c.drawImage(refImage,0,0,w,h);};refImage.src=row.reference_image;}
    if($('labels').checked)row.annotations.forEach((a,i)=>{
      const font=Math.max(16,Math.min(36,14*w/Math.max(250,canvas.clientWidth)));c.font='bold '+font+'px Segoe UI';
      const tw=c.measureText(a.label).width+12,x=Math.min(w-tw-3,Math.max(3,a.x*w/256)),y=Math.min(h-8,Math.max(font+4,a.y*h/256+(i%2)*18));
      c.fillStyle='rgba(0,0,0,.80)';c.fillRect(x,y-font,tw,font+7);c.fillStyle=colors[a.region];c.fillText(a.label,x+6,y);
    });
  };image.src=row.image;
  $('fragment-list').innerHTML=row.annotations.length?row.annotations.map(a=>'<div class="fragment"><span style="color:'+colors[a.region]+'">'+a.label+'<br><small>'+names[a.region]+'</small></span><small>'+(a.contact_26?'Contacto 26':'')+'</small></div>').join(''):'<p>No hay instancias predichas en este corte.</p>';
}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>showTab(b.dataset.tab));
$('case').onchange=()=>loadCase().catch(showError);$('slice').oninput=draw;$('labels').onchange=draw;
$('reference').onchange=draw;$('anatomy').onchange=draw;
$('prev').onclick=()=>{$('slice').value=Math.max(0,Number($('slice').value)-1);draw();};
$('next').onclick=()=>{$('slice').value=Math.min(slices.length-1,Number($('slice').value)+1);draw();};
function showError(error){$('scope').textContent='No se pudieron cargar los resultados. Inicia el servidor local indicado en el README.';console.error(error);}
fetch('data/catalogo.json').then(r=>{if(!r.ok)throw Error('No se encuentra el catálogo');return r.json();}).then(data=>{catalog=data;$('case').innerHTML=data.map(c=>'<option value="'+c.id+'">Caso '+c.id+'</option>').join('');return loadCase();}).catch(showError);
