import unittest
import numpy as np
import torch
from pengwin.instancias_estables import reconstruir
from pengwin.instances import separation_distances,match_instances
from pengwin.models.ablacion import crear_ablacion
from pengwin.giou import giou_aligned,grid_giou_loss

class CierreTests(unittest.TestCase):
    def test_giou_identity_and_disjoint_gradients(self):
        box=torch.tensor([[.1,.1,.3,.3]])
        self.assertAlmostEqual(float(giou_aligned(box,box)),1.)
        pred=torch.tensor([[.6,.6,.8,.8]],requires_grad=True)
        loss=(1-giou_aligned(pred,box)).mean();loss.backward()
        self.assertGreater(float(loss.detach()),1.);self.assertGreater(float(pred.grad.abs().sum()),0)
        grid=torch.randn(1,8,16,16,requires_grad=True)
        loss=grid_giou_loss(grid,torch.tensor([[0,0,.1,.1,.3,.3]]));loss.backward()
        self.assertTrue(torch.isfinite(grid.grad).all())
    def test_interfaces_separate_touching_fragments(self):
        sem=np.zeros((7,9,12),np.uint8);sem[1:6,1:8,1:11]=1
        edge=np.zeros_like(sem,float);edge[:,:,5:7]=1
        pred,m=reconstruir(sem,edge,np.ones_like(edge),(1,1,1),{'method':'interfaces','threshold':.5,'min_volume_mm3':0,'seed_mm3':0})
        self.assertEqual(len(m),2);self.assertTrue(np.all(pred[sem>0]>0))
    def test_volume_filter_uses_physical_spacing(self):
        sem=np.zeros((5,5,5),np.uint8);sem[2,2,2]=1
        a,_=reconstruir(sem,sem*0,sem*0,(1,1,1),{'method':'componentes','min_volume_mm3':2})
        b,_=reconstruir(sem,sem*0,sem*0,(2,2,2),{'method':'componentes','min_volume_mm3':2})
        self.assertEqual(a.max(),0);self.assertEqual(b.max(),1)
    def test_oracle_distance_comparison_is_zero(self):
        gt=np.zeros((5,5,12),np.int32);gt[1:4,1:4,1:4]=1;gt[1:4,1:4,7:9]=2
        a=separation_distances(gt,{1:1,2:1},(2,1,.5),principal_ids={1:1})
        b=separation_distances(gt,{1:1,2:1},(2,1,.5))
        self.assertEqual(a[0]['distance_surface_voxel_centers_mm'],2.)
        self.assertEqual(a[0]['distance_surface_voxel_centers_mm']-b[0]['distance_surface_voxel_centers_mm'],0)
        self.assertTrue(all(r['iou']==1 for r in match_instances(gt,gt,{1:1,2:1})))
    def test_resnet_own_heads_shape(self):
        torch.set_num_threads(2);m=crear_ablacion('resnet_sin_transfer');m.eval()
        with torch.no_grad():out=m(torch.zeros(1,3,256,256))
        self.assertEqual(tuple(out['grid'].shape),(1,8,16,16));self.assertEqual(tuple(out['mascaras'].shape),(1,4,256,256))
        self.assertEqual(m.segmentation_head.skips[0].out_channels,8)
if __name__=='__main__':unittest.main()
