import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np,torch
from pengwin.dataset import extraer_bboxes_region
from pengwin.models.detector import PelvisDetector
from pengwin.instances import fragment_boundaries,separation_distances,reconstruct_instances,match_instances
from entrenar_pacientes import seg_loss
from pengwin.models.loss import PelvisDetectionLoss
from pengwin.metrics_avance3 import detection_metrics,classification_metrics
from inferir_fragmentos_3d import reference_grid
import SimpleITK as sitk

class Tests(unittest.TestCase):
    def test_physical_grid_center_preserved(self):
        image=sitk.Image([512,384,5],sitk.sitkFloat32)
        image.SetSpacing((.7,.8,2.5));image.SetOrigin((15.,-50.,20.))
        image.SetDirection((0.,-1.,0.,1.,0.,0.,0.,0.,1.))
        ref=reference_grid(image)
        np.testing.assert_allclose(image.TransformContinuousIndexToPhysicalPoint((255.5,191.5,2.)),ref.TransformContinuousIndexToPhysicalPoint((127.5,127.5,2.)))
        np.testing.assert_allclose(ref.GetSpacing(),(1.4,1.2,2.5))

    def test_ap_penalizes_missed_and_false_detections(self):
        record={'gt':[[0,0,0,1,1]],'boxes':[[0,0,1,1]],'scores':[.9],'labels':[0]}
        self.assertAlmostEqual(detection_metrics([record])['mAP50'],1.)
        empty={'gt':[],'boxes':[[0,0,1,1]],'scores':[.99],'labels':[0]}
        self.assertAlmostEqual(detection_metrics([record,empty])['mAP50'],.5)
        record['boxes']=[];record['scores']=[];record['labels']=[]
        self.assertEqual(detection_metrics([record])['mAP50'],0.)

    def test_auc_ties_and_unavailable_class(self):
        result=classification_metrics([[1,0,1],[0,0,0]],[[.5,.2,.9],[.5,.1,.1]])
        self.assertEqual(result['per_class'][0]['auc'],.5)
        self.assertIsNone(result['per_class'][1]['auc'])
        self.assertEqual(result['per_class'][2]['auc'],1.)

    def test_gt_principal_does_not_change_with_volume(self):
        ids=np.zeros((3,3,5),np.int32);ids[1,1,0]=1;ids[1,1,3:]=2
        row=separation_distances(ids,{1:1,2:1},(1,1,1),{1:1})[0]
        self.assertEqual(row['principal'],1);self.assertEqual(row['instance'],2)

    def test_small_gt_is_preserved(self):
        a=np.zeros((12,12),np.uint8);a[1,1]=1
        self.assertEqual(len(extraer_bboxes_region(a)),1)
        self.assertEqual(extraer_bboxes_region(np.zeros_like(a)),[])
        self.assertEqual(extraer_bboxes_region(a,min_pixeles=15),[])

    def test_filter_is_region_area_not_components(self):
        a=np.zeros((12,12),np.uint8);a[2:8,2:8]=1;a[11,11]=1
        self.assertEqual(extraer_bboxes_region(a,min_pixeles=15)[0]['bbox'][2:], [1.,1.])

    def test_interfaces_invariant_to_ids(self):
        a=np.array([[1,1,2,2],[1,1,2,2]],np.uint8);b=a.copy();b[a==1]=5;b[a==2]=8
        np.testing.assert_array_equal(fragment_boundaries(a),fragment_boundaries(b))
        self.assertEqual(fragment_boundaries(a).sum(),4)

    def test_balanced_loss_and_three_heads_backward(self):
        torch.set_num_threads(2);model=PelvisDetector(pretrained=False)
        x=torch.rand(2,3,256,256);out=model(x)
        ids=torch.zeros(2,256,256,dtype=torch.long);ids[:,80:130,80:130]=1;edge=torch.zeros(2,256,256);edge[:,100,80:130]=1
        boxes=torch.tensor([[0,0,.3,.3,.5,.5]],dtype=torch.float32)
        ld=PelvisDetectionLoss(balancear_obj=True)(out,boxes,torch.tensor([[1.,0,0],[0,0,0]]))['loss_total']
        ls,_=seg_loss(out,ids,edge);loss=ld+ls;loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(out['mascaras'].shape,(2,4,256,256))
        self.assertEqual(model.segmentation_head.latent_channels,8)
        self.assertGreater(model.segmentation_head.bordes_fragmento.weight.grad.abs().sum(),0)

    def test_edt_spacing_and_contact(self):
        a=np.zeros((3,3,3),np.int32);a[0,0,0]=1;a[1,1,1]=2
        row=separation_distances(a,{1:1,2:1},(3,2,1))[0]
        self.assertAlmostEqual(row['distance_surface_voxel_centers_mm'],np.sqrt(14));self.assertTrue(row['contact_26'])

    def test_watershed_and_matching(self):
        sem=np.ones((3,6,10),np.uint8);edge=np.zeros_like(sem,float);edge[:,:,4:6]=1
        pred,mapping=reconstruct_instances(sem,edge,(1,1,1),min_volume_mm3=0)
        self.assertEqual(len(mapping),2)
        rows=match_instances(pred,pred.astype(np.uint8),mapping)
        self.assertTrue(all(r['dice']==1 for r in rows))
        self.assertEqual(match_instances(np.zeros_like(pred),pred.astype(np.uint8),{})[0]['dice'],0.)

if __name__=='__main__':unittest.main()
