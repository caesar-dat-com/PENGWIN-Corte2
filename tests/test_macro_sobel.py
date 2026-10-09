import unittest
import numpy as np
import torch
import SimpleITK as sitk
from pengwin.refinamiento_macro import sobel_tensor,perdida_macro_contorno,refinar_probabilidades,control_calidad
from pengwin.io import normalizar_lps,verificar_alineacion


class MacroTests(unittest.TestCase):
    def test_sobel_constant_and_step(self):
        x=torch.ones(1,1,8,8)
        a,b=sobel_tensor(x);self.assertEqual(float(a.abs().sum()+b.abs().sum()),0)
        x[:,:,:,4:]=0;a,b=sobel_tensor(x)
        self.assertGreater(float(a.abs().sum()),0);self.assertEqual(float(b.abs().sum()),0)

    def test_loss_empty_and_foreground_gradients(self):
        for foreground in (False,True):
            logits=torch.randn(2,4,8,8,requires_grad=True);target=torch.zeros(2,8,8,dtype=torch.long)
            if foreground:target[:,2:6,2:6]=1
            m,c=perdida_macro_contorno(logits,target);(m+c).backward()
            self.assertTrue(torch.isfinite(logits.grad).all());self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_refinement_identity_and_confident_island(self):
        p=np.zeros((4,9,9),np.float32);p[0]=.99;p[1]=.01;p[:,4,4]=[.01,.99,0,0]
        im=np.zeros((9,9),np.float32)
        np.testing.assert_array_equal(refinar_probabilidades(p,im,strength=0),p)
        r=refinar_probabilidades(p,im,strength=.5)
        np.testing.assert_allclose(r.sum(0),1);self.assertEqual(r[:,4,4].argmax(),1)

    def test_refinement_respects_intensity_edge(self):
        p=np.zeros((4,9,9),np.float32);p[0]=.6;p[1]=.4;p[0,:,5:]=.4;p[1,:,5:]=.6
        im=np.zeros((9,9),np.float32);im[:,5:]=1
        r=refinar_probabilidades(p,im,strength=.5,sigma=.05)
        np.testing.assert_allclose(r,p,atol=1e-6)

    def test_lps_landmark_and_roundtrip(self):
        a=np.arange(3*4*5,dtype=np.int16).reshape(3,4,5)
        image=sitk.GetImageFromArray(a);image.SetSpacing((.7,.9,2.));image.SetOrigin((10,20,30))
        image.SetDirection(sitk.DICOMOrientImageFilter_GetDirectionCosinesFromOrientation('RAI'))
        oriented=normalizar_lps(image)
        self.assertEqual(sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(oriented.GetDirection()),'LPS')
        point=image.TransformIndexToPhysicalPoint((1,2,1));idx=oriented.TransformPhysicalPointToIndex(point)
        self.assertEqual(image[1,2,1],oriented[idx])
        back=sitk.DICOMOrient(oriented,'RAI');verificar_alineacion(image,back,'test')
        np.testing.assert_array_equal(sitk.GetArrayFromImage(back),a)

    def test_qc_does_not_delete_fragments(self):
        sem=np.zeros((5,20,20),np.uint8);sem[2,::3,::3]=1;before=sem.copy()
        q=control_calidad(sem);self.assertGreater(q['regions'][0]['components'],10)
        self.assertTrue(q['alerts']);np.testing.assert_array_equal(sem,before)

if __name__=='__main__':unittest.main()
