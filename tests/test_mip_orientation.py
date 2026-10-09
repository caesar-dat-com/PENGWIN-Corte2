import unittest
import numpy as np
from pengwin.viz_mip import figura_mip


class MIPOrientationTests(unittest.TestCase):
    def test_superior_landmark_is_at_top_in_every_rotation(self):
        volume = np.zeros((5,7,7), np.float32)
        volume[0,3,3] = 400
        volume[-1,3,3] = 1200
        fig = figura_mip(volume,(1,1,1),'synthetic',mm=1,paso_grados=90)
        self.assertEqual(fig.layout.yaxis.autorange, 'reversed')
        for frame in fig.frames:
            projection=np.asarray(frame.data[0].z)
            self.assertGreater(projection[0].max(),projection[-1].max())
