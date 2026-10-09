import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from pengwin.models.sesion2 import PelvisSesion2
from pengwin.fragmentos_sesion2 import objetivos_fragmentos,reconstruir_fragmentos
from entrenar_sesion2 import seg_loss


class Sesion2Tests(unittest.TestCase):
    def test_local_ids_and_individual_masks(self):
        ids=np.zeros((16,16),np.uint8);ids[2:10,2:6]=1;ids[2:10,6:10]=2
        targets=objetivos_fragmentos(ids)
        self.assertEqual(targets['masks'].shape,(2,16,16))
        self.assertEqual(targets['labels'].tolist(),[1,1])
        perm=ids.copy();perm[ids==1]=7;perm[ids==2]=5
        np.testing.assert_allclose(targets['interiores'],objetivos_fragmentos(perm)['interiores'])
        self.assertEqual(objetivos_fragmentos(np.zeros_like(ids))['masks'].shape,(0,16,16))

    def test_skips_receive_gradients_and_keep_eight_channels(self):
        torch.set_num_threads(2);m=PelvisSesion2().train()
        shapes=[]
        hooks=[layer.register_forward_pre_hook(lambda mod,args:shapes.append(args[0].shape[1])) for layer in (m.segmentation_head.up1.conv,m.segmentation_head.up2.conv,m.segmentation_head.up3.conv,m.segmentation_head.up4.conv)]
        x=torch.randn(2,3,128,128);out=m(x)
        target=torch.zeros(2,128,128,dtype=torch.long);target[:,30:80,30:80]=1
        loss,_=seg_loss(out,target,torch.zeros_like(target).float(),torch.zeros_like(target).float())
        loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(shapes,[8,8,8,8])
        self.assertEqual(len(out['auxiliares']),3)
        self.assertTrue(all(p.weight.grad.abs().sum()>0 for p in m.segmentation_head.skips))
        for h in hooks:h.remove()
        m.eval()
        with torch.no_grad():self.assertEqual(m(x)['auxiliares'],[])

    def test_touching_fragments_separated_by_interiors(self):
        ids=np.zeros((24,24),np.uint8);ids[3:21,3:12]=1;ids[3:21,12:21]=2
        cores=objetivos_fragmentos(ids)['interiores']
        sem=np.repeat((ids>0)[None],3,axis=0).astype(np.uint8)
        pred,mapping=reconstruir_fragmentos(sem,np.zeros_like(sem,float),np.repeat(cores[None],3,0),(1,1,1),min_volume_mm3=0)
        self.assertEqual(len(mapping),2)
        self.assertNotEqual(pred[1,12,6],pred[1,12,18])

    def test_seed_filter_keeps_isolated_component_without_seed(self):
        sem=np.zeros((3,12,12),np.uint8);sem[:,2:5,2:5]=1
        cores=np.zeros_like(sem,float);cores[:,3,3]=1
        pred,mapping=reconstruir_fragmentos(sem,np.zeros_like(cores),cores,(1,1,1),min_volume_mm3=0,seed_min_volume_mm3=50)
        self.assertEqual(len(mapping),1)
        np.testing.assert_array_equal(pred>0,sem>0)

if __name__=='__main__':unittest.main()
