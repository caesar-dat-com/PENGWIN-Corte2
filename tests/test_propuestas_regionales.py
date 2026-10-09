import unittest
import numpy as np
from pengwin.propuestas_regionales import combinar_regiones


class FusionTests(unittest.TestCase):
    def test_rpn_priority_and_missing_region_fallback(self):
        boxes, scores, labels, sources = combinar_regiones(
            [[.1,.1,.4,.4],[.2,.2,.7,.7]], [.9,.8], [0,1],
            [[.15,.15,.45,.45],[.1,.1,.5,.5],[.3,.3,.6,.6]],
            [.7,.6,.09], [0,0,2], padding=0)
        self.assertEqual(labels.tolist(), [0,1])
        self.assertEqual(sources, ['rpn_grid','grid_respaldo'])
        np.testing.assert_allclose(boxes[0], [.15,.15,.45,.45])
        np.testing.assert_allclose(scores, [.7,.8])

    def test_empty_predictions_do_not_create_boxes(self):
        boxes, scores, labels, sources = combinar_regiones([],[],[],[],[],[])
        self.assertEqual(boxes.shape, (0,4))
        self.assertEqual(len(labels), 0)
        self.assertEqual(sources, [])

    def test_padding_stays_inside_image(self):
        boxes, _, _, _ = combinar_regiones([],[],[],[[0,0,1,1]],[.8],[2],padding=.02)
        np.testing.assert_allclose(boxes, [[0,0,1,1]])
