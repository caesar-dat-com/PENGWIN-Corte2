import unittest
import torch
from pengwin.models.rpn_grid import RPNGridHead,encode_boxes,decode_boxes,targets_for_anchors,proposal_loss,decode_proposals

class ProposalTests(unittest.TestCase):
    def test_box_encoding_roundtrip(self):
        anchors=torch.tensor([[.1,.2,.4,.5],[.4,.4,.7,.8]])
        boxes=torch.tensor([[.05,.15,.5,.55],[.45,.3,.65,.95]])
        torch.testing.assert_close(decode_boxes(encode_boxes(boxes,anchors),anchors),boxes)

    def test_forced_match_keeps_small_gt(self):
        head=RPNGridHead([[.2,.2],[.4,.3]])
        anchors=head.anchors(4,4,'cpu');gt=torch.tensor([[.49,.49,.50,.50],[.1,.1,.11,.11]])
        obj,match=targets_for_anchors(anchors,gt)
        self.assertEqual(set(match[obj==1].tolist()),{0,1})

    def test_empty_and_positive_targets_have_gradients(self):
        torch.set_num_threads(2);head=RPNGridHead([[.2,.2],[.4,.3]],in_channels=8)
        grid=head(torch.rand(2,8,4,4));loss=proposal_loss(head,grid,[torch.empty(0,5),torch.tensor([[1,.2,.2,.6,.7]])]);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(head.output.weight.grad.abs().sum()),0)
        boxes,scores,labels=decode_proposals(head,grid,conf=0,one_per_region=True)
        self.assertEqual(len(labels),len(labels.unique()));self.assertTrue(((boxes>=0)&(boxes<=1)).all())

if __name__=='__main__':unittest.main()
