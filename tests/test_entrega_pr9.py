import unittest
import tempfile
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from pengwin.evaluacion_distancias import comparar_distancias
from comprobar_paquete import comprobar,sha256

class Entrega(unittest.TestCase):
    def test_distance_threshold_and_missing_pairs(self):
        matches=[{'pred':1,'gt':1,'iou':.9},{'pred':2,'gt':2,'iou':.2}]
        pred=[{'instance':2,'principal':1,'distance_surface_voxel_centers_mm':4.}]
        gt=[{'instance':2,'principal':1,'distance_surface_voxel_centers_mm':2.}]
        self.assertEqual(comparar_distancias(matches,pred,gt,.1)['distance_MAE_mm'],2.)
        strict=comparar_distancias(matches,pred,gt,.5)
        self.assertEqual(strict['distance_valid_pairs'],0);self.assertIsNone(strict['distance_MAE_mm'])
    def test_principal_also_requires_match(self):
        matches=[{'pred':2,'gt':2,'iou':1.}]
        row=[{'instance':2,'principal':1,'distance_surface_voxel_centers_mm':0.}]
        self.assertIsNone(comparar_distancias(matches,row,row,.5)['distance_MAE_mm'])
    def test_package_missing_and_checkpoint_mismatch(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root=Path(tmp);ck=root/'best.pth';ck.write_bytes(b'fixture')
            errors=comprobar(root/'dashboard',ck,'0'*64)
            self.assertTrue(any('SHA256' in e for e in errors))
            self.assertTrue(any('catalogo' in e for e in errors))
    def test_complete_package(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root=Path(tmp);ck=root/'best.pth';ck.write_bytes(b'fixture');dash=root/'dashboard'
            for name in ['index.html','app.js','style.css','assets/plotly.min.js','assets/informe.md','data/002/mip.html','data/002/malla.html','data/002/a.png']:
                p=dash/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
            (dash/'data/catalogo.json').write_text(json.dumps([{'id':'002','slices':1,'checkpoint_sha256':sha256(ck)}]))
            (dash/'data/002/cortes.json').write_text(json.dumps([{'image':'data/002/a.png'}]))
            self.assertEqual(comprobar(dash,ck,sha256(ck)),[])

if __name__=='__main__': unittest.main()
