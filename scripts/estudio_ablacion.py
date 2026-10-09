"""Cuatro ejecuciones controladas desde cabezas aleatorias, 3 épocas, train/val."""
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from torch.utils.data import DataLoader
from entrenar_sesion2 import Samples,collate,seg_loss,evaluate
from pengwin.models.ablacion import crear_ablacion
from pengwin.models.loss import PelvisDetectionLoss

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def main():
    torch.set_num_threads(4);dest=ROOT/'salidas/cierre/ablaciones';dest.mkdir(parents=True,exist_ok=True)
    train,val=Samples(ROOT/'salidas/revision_oct08/datos','train'),Samples(ROOT/'salidas/revision_oct08/datos','val')
    vl=DataLoader(val,batch_size=8,collate_fn=collate);rows=[]
    pretrained=ROOT/'pesos_externos/resnet18-f37072fd.pth'
    if not pretrained.exists():raise ValueError('Faltan pesos oficiales ImageNet del backbone ResNet18')
    for name in ('fundidora_cbam','fundidora_sin_cbam','resnet_sin_transfer','resnet_transfer'):
        out=dest/name
        if (out/'resultado.json').exists():rows.append(json.loads((out/'resultado.json').read_text(encoding='utf-8')));continue
        out.mkdir(exist_ok=True);torch.manual_seed(42);np.random.seed(42)
        model=crear_ablacion(name,pretrained)
        head_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for k,v in model.state_dict().items() if k.startswith(('detection_head.','classification_head.','segmentation_head.')))).hexdigest()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);criterion=PelvisDetectionLoss(balancear_obj=True)
        tl=DataLoader(train,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(42),collate_fn=collate)
        config={'name':name,'epochs':3,'seed':42,'train_slices':len(train),'val_slices':len(val),'train_patients':70,'val_patients':15,'lr':3e-4,'batch':8,'heads_initial_sha256':head_hash,'pretrained_backbone_sha256':hashlib.sha256(pretrained.read_bytes()).hexdigest() if name=='resnet_transfer' else None,'test_used':False,'status':'running'}
        save(out/'config.json',config);history=[]
        for epoch in range(1,4):
            start=time.perf_counter();model.train();total=0
            for b in tl:
                optimizer.zero_grad(set_to_none=True);pred=model(b['x']);loss=criterion(pred,b['boxes'],b['classes'])['loss_total']
                ls,_=seg_loss(pred,b['semantic'],b['boundary'],b['cores']);loss=loss+1.2*ls
                if not torch.isfinite(loss):raise ValueError('Loss no finita')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=float(loss.detach())*len(b['x'])
            metrics,_=evaluate(model,vl,torch.device('cpu'));history.append({'epoch':epoch,'loss':total/len(train),'seconds':time.perf_counter()-start,'val':metrics});save(out/'historial.json',history)
            print('ABLACION',name,epoch,'Dice',round(metrics['semantic']['macro_dice'],4),'mAP50',round(metrics['detection']['mAP50'],4),flush=True)
        config['status']='completed';save(out/'config.json',config)
        torch.save({'model':model.state_dict(),'config':config,'epoch':3,'val':metrics},out/'final.pth')
        result={'name':name,'config':config,'metrics':metrics};save(out/'resultado.json',result);rows.append(result)
    assert rows[0]['config']['heads_initial_sha256']==rows[1]['config']['heads_initial_sha256']
    assert rows[2]['config']['heads_initial_sha256']==rows[3]['config']['heads_initial_sha256']
    save(dest/'comparacion.json',{'scope':'Estudio corto de 3 epocas iguales por par; no prueba convergencia ni superioridad universal. CBAM se compara dentro de Fundidora; transfer se compara dentro de ResNet18. No usar diferencia Fundidora/ResNet como efecto de transferencia. No se selecciona el modelo final con test.','results':rows})
if __name__=='__main__':main()
