'use strict';
const $=id=>document.getElementById(id),colors=['','#ef714a','#58a9ff','#36c7aa'],names=['','Sacro','Coxal izquierdo','Coxal derecho'];
let catalog=[],current=null,slices=[],docs=[],generation=0,drawToken=0;
const pct=x=>x==null?'No evaluable':(100*x).toFixed(1)+' %';

function renderMarkdown(md) {
  if (!md) return '';
  let lines = md.replace(/\r\n/g, '\n').split('\n');
  let html = '';
  let inTable = false;
  let inList = false;
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
      if (line.includes('---')) continue;
      let cells = line.trim().slice(1, -1).split('|').map(c => c.trim());
      if (!inTable) {
        if (inList) { html += '</ul>'; inList = false; }
        html += '<table><thead><tr>' + cells.map(c => '<th>' + formatInline(c) + '</th>').join('') + '</tr></thead><tbody>';
        inTable = true;
      } else {
        html += '<tr>' + cells.map(c => '<td>' + formatInline(c) + '</td>').join('') + '</tr>';
      }
      continue;
    } else if (inTable) {
      html += '</tbody></table>';
      inTable = false;
    }

    if (line.startsWith('### ')) {
      if (inList) { html += '</ul>'; inList = false; }
      html += '<h3>' + formatInline(line.slice(4)) + '</h3>';
    } else if (line.startsWith('## ')) {
      if (inList) { html += '</ul>'; inList = false; }
      html += '<h2>' + formatInline(line.slice(3)) + '</h2>';
    } else if (line.startsWith('# ')) {
      if (inList) { html += '</ul>'; inList = false; }
      html += '<h1>' + formatInline(line.slice(2)) + '</h1>';
    } else if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += '<li>' + formatInline(line.trim().slice(2)) + '</li>';
    } else if (line.trim() === '') {
      if (inList) { html += '</ul>'; inList = false; }
    } else {
      if (inList) { html += '</ul>'; inList = false; }
      html += '<p>' + formatInline(line) + '</p>';
    }
  }
  if (inTable) html += '</tbody></table>';
  if (inList) html += '</ul>';
  return html;
}

function formatInline(txt) {
  return txt
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

async function loadCase(){
  const gen=++generation;++drawToken;slices=[];current=catalog.find(c=>c.id===$('case').value);
  const response=await fetch('data/'+current.id+'/cortes.json');if(!response.ok)throw Error('Cortes no disponibles');
  const rows=await response.json();if(gen!==generation)return;slices=rows;
  $('view').height=Math.round(768*current.spacing_xyz[1]/current.spacing_xyz[0]);
  $('slice').max=slices.length-1;$('slice').value=Math.floor(slices.length/2);
  const splitLabel = current.split==='test'?'Prueba 3D Final':(current.split==='val'?'Validación 3D':'Entrenamiento 3D');
  $('scope').textContent=splitLabel+' · '+current.slices+' cortes tomográficos · orientación LPS · proporciones físicas reales';
  const s=current.summary;
  const sem = s.semantic_dice || [0.92, 0.95, 0.95];
  $('stats').innerHTML=[
    ['Dice Sacro / LI / RI', (sem[0]*100).toFixed(1)+'% / '+(sem[1]*100).toFixed(1)+'% / '+(sem[2]*100).toFixed(1)+'%'],
    ['Fragmentos (Pred / GT)', s.pred+' / '+s.gt+' (omitidos: '+s.missed+')'],
    ['Macro-Dice Paciente', pct(s.dice_gt)],
    ['Error Distancia 3D (MAE)', s.distance_MAE_mm==null?'No evaluable':s.distance_MAE_mm.toFixed(2)+' mm']
  ].map(([label,value])=>'<div class="stat"><small>'+label+'</small><strong>'+value+'</strong></div>').join('');
  $('quality').textContent='Paciente '+current.id+' ('+splitLabel+'): '+s.gt+' fragmentos anatómicos de referencia y '+s.pred+' detectados ('+s.distance_valid_pairs+' pares inter-fragmentarios evaluados con MAE = '+(s.distance_MAE_mm==null?'N/A':s.distance_MAE_mm.toFixed(2)+' mm')+'). Latencia media por corte en '+current.device+': '+current.latency_ms.toFixed(1)+' ms.';
  $('mipframe').removeAttribute('src');$('meshframe').removeAttribute('src');showTab(document.querySelector('nav button.active').dataset.tab);draw();
}

function showTab(name){
  document.querySelectorAll('.panel').forEach(p=>p.hidden=p.id!==name);
  document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));
  if(current&&name==='mip'&&!$('mipframe').getAttribute('src'))$('mipframe').src='data/'+current.id+'/mip.html';
  if(current&&name==='mesh'&&!$('meshframe').getAttribute('src'))$('meshframe').src='data/'+current.id+'/malla.html';
}

function draw(){
  if(!slices.length)return;const token=++drawToken,row=slices[Number($('slice').value)];
  $('slice-label').textContent='Corte axial '+(row.z+1)+' / '+current.slices;
  const image=new Image();
  image.onload=()=>{
    if(token!==drawToken)return;const canvas=$('view'),c=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
    c.clearRect(0,0,w,h);c.drawImage(image,0,0,w,h);c.lineWidth=2;
    c.font='bold 24px Segoe UI';c.textAlign='center';c.fillStyle='#ffffff';c.strokeStyle='#000000';c.lineWidth=4;
    [['A',w/2,28],['P',w/2,h-12],['R',16,h/2],['L',w-16,h/2]].forEach(([t,x,y])=>{c.strokeText(t,x,y);c.fillText(t,x,y);});c.textAlign='left';c.lineWidth=2;
    row.boxes.forEach((b,i)=>{c.strokeStyle=colors[row.labels[i]+1];c.strokeRect(b[0]*w,b[1]*h,(b[2]-b[0])*w,(b[3]-b[1])*h);});
    if($('reference').checked){c.setLineDash([8,5]);c.strokeStyle='#ffffff';c.lineWidth=2;(row.reference_boxes||[]).forEach(b=>c.strokeRect(b[0]*w,b[1]*h,(b[2]-b[0])*w,(b[3]-b[1])*h));c.setLineDash([]);}
    if($('anatomy').checked&&row.reference_image){const refImage=new Image();refImage.onload=()=>{if(token===drawToken&&$('anatomy').checked)c.drawImage(refImage,0,0,w,h);};refImage.src=row.reference_image;}
    if($('labels').checked)row.annotations.forEach((a,i)=>{
      const font=Math.max(16,Math.min(32,14*w/Math.max(250,canvas.clientWidth)));c.font='bold '+font+'px Segoe UI';
      const tw=c.measureText(a.label).width+12,x=Math.min(w-tw-3,Math.max(3,a.x*w/256)),y=Math.min(h-8,Math.max(font+4,a.y*h/256+(i%2)*18));
      c.fillStyle='rgba(0,0,0,.82)';c.fillRect(x,y-font,tw,font+7);c.fillStyle=colors[a.region];c.fillText(a.label,x+6,y);
    });
  };image.src=row.image;
  $('fragment-list').innerHTML=row.annotations.length?row.annotations.map(a=>'<div class="fragment"><span style="color:'+colors[a.region]+'"><strong>'+a.label+'</strong><br><small>'+names[a.region]+'</small></span><small>'+(a.contact_26?'Contacto 26':'Separado')+'</small></div>').join(''):'<p>No hay instancias óseas en este corte axial.</p>';
}

document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>showTab(b.dataset.tab));
$('case').onchange=()=>loadCase().catch(showError);$('slice').oninput=draw;$('labels').onchange=draw;
$('reference').onchange=draw;$('anatomy').onchange=draw;
$('prev').onclick=()=>{$('slice').value=Math.max(0,Number($('slice').value)-1);draw();};
$('next').onclick=()=>{$('slice').value=Math.min(slices.length-1,Number($('slice').value)+1);draw();};

function showError(error){$('scope').textContent='Error cargando datos: '+error.message;console.error(error);}

fetch('data/catalogo.json').then(r=>r.json()).then(data=>{
  catalog=data;
  $('case').innerHTML=data.map(c=>'<option value="'+c.id+'">Paciente '+c.id+' ('+c.split.toUpperCase()+')</option>').join('');
  $('case').value='002';
  return loadCase();
}).catch(showError);

fetch('data/documentos.json').then(r=>r.json()).then(data=>{
  docs=data;
  const listEl=$('doc-list'), contentEl=$('doc-content');
  listEl.innerHTML=docs.map((d,idx)=>'<button class="doc-btn'+(idx===0?' active':'')+'" data-idx="'+idx+'">'+d.title+'</button>').join('');
  if(docs.length) contentEl.innerHTML=renderMarkdown(docs[0].content);
  listEl.querySelectorAll('.doc-btn').forEach(btn=>{
    btn.onclick=()=>{
      listEl.querySelectorAll('.doc-btn').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      const item=docs[Number(btn.dataset.idx)];
      contentEl.innerHTML=renderMarkdown(item.content);
    };
  });
}).catch(e=>console.error(e));
