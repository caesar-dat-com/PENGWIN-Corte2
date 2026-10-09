import unittest
import torch
from pengwin.negativos_dificiles import perdida_negativos

class HardNegativeTests(unittest.TestCase):
    def test_never_penalizes_annotated_foreground(self):
        logits=torch.randn(1,4,4,4,requires_grad=True)
        loss=perdida_negativos(logits,torch.ones(1,4,4,dtype=torch.long),torch.ones(1,3,4,4))
        loss.backward();self.assertEqual(loss.item(),0);self.assertEqual(logits.grad.abs().sum().item(),0)

    def test_false_positive_receives_background_gradient(self):
        logits=torch.tensor([0.,4.,0.,0.]).reshape(1,4,1,1).requires_grad_()
        loss=perdida_negativos(logits,torch.zeros(1,1,1,dtype=torch.long),torch.ones(1,3,1,1))
        loss.backward();self.assertLess(logits.grad[0,0,0,0].item(),0);self.assertGreater(logits.grad[0,1,0,0].item(),0)
