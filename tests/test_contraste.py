import unittest
import torch
from pengwin.contraste import clahe_batch

class ContrastTests(unittest.TestCase):
    def test_preserves_input_and_channel_alignment(self):
        original=torch.linspace(0,1,64*64).reshape(1,1,64,64).repeat(2,3,1,1)
        saved=original.clone();result=clahe_batch(original)
        torch.testing.assert_close(original,saved)
        self.assertEqual(result.shape,original.shape)
        self.assertEqual(result.dtype,torch.float32)
        self.assertTrue(((result>=0)&(result<=1)).all())
        torch.testing.assert_close(result[:,0],result[:,1])
        torch.testing.assert_close(result,clahe_batch(original))
