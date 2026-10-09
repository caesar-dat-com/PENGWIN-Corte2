import unittest
import numpy as np
from pengwin.mascara_corporal import mascara_corporal

class BodyMaskTests(unittest.TestCase):
    def test_body_encloses_cavity_and_excludes_detached_table(self):
        ct=np.full((8,50,50),-1000.,np.float32)
        ct[:,8:35,8:40]=0
        ct[:,16:22,16:22]=-1000
        ct[:,43:46,:]=300
        body=mascara_corporal(ct,(2,1,1),opening_mm=2,margin_mm=2)
        self.assertTrue(body[:,18,18].all())
        self.assertFalse(body[:,44,:].any())
        self.assertTrue(body[:,9:34,9:39].all())

    def test_empty_ct_stays_empty(self):
        self.assertFalse(mascara_corporal(np.full((3,9,9),-1000),(1,1,1)).any())

    def test_invalid_spacing_rejected(self):
        with self.assertRaises(ValueError):mascara_corporal(np.zeros((3,3,3)),(1,0,1))
