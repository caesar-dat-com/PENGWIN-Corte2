"""Comprueba recursos y checkpoint antes de crear una entrega transportable."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def comprobar(dashboard, checkpoint, expected):
    dashboard = dashboard.resolve()
    problems = []
    def resource(relative):
        path = (dashboard / relative).resolve()
        if not path.is_relative_to(dashboard):
            problems.append('Ruta fuera del dashboard: ' + relative)
            return None
        if not path.is_file():
            problems.append('Falta recurso: ' + relative)
            return None
        return path
    for name in ('index.html','app.js','style.css','assets/plotly.min.js','assets/informe.md'):
        resource(name)
    catalog_path = resource('data/catalogo.json')
    if catalog_path:
        catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
        if not catalog: problems.append('Catálogo vacío')
        for case in catalog:
            if case.get('checkpoint_sha256') != expected:
                problems.append('Checkpoint del catálogo no identificado o distinto: ' + case['id'])
            folder = 'data/' + case['id'] + '/'
            rows_path = resource(folder + 'cortes.json')
            resource(folder + 'mip.html'); resource(folder + 'malla.html')
            if rows_path:
                rows = json.loads(rows_path.read_text(encoding='utf-8'))
                if len(rows) != case['slices']: problems.append('Número de cortes distinto: ' + case['id'])
                for row in rows:
                    resource(row['image'])
                    if row.get('reference_image'): resource(row['reference_image'])
    if not checkpoint.is_file():
        problems.append('Falta checkpoint: ' + str(checkpoint))
    elif sha256(checkpoint) != expected:
        problems.append('SHA256 del checkpoint no coincide')
    return problems

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dashboard',type=Path,default=ROOT/'dashboard')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'salidas/fix_huesos_v1/best.pth')
    parser.add_argument('--package',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    expected=json.loads((ROOT/'salidas/fix_huesos_v1/best_pth_sha256.json').read_text())['sha256']
    issues=comprobar(args.dashboard,args.checkpoint,expected)
    report={'ready':not issues,'expected_checkpoint_sha256':expected,'issues':issues}
    if args.report: args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('LISTO' if not issues else f'PENDIENTE: {len(issues)} incidencias')
    for issue in issues[:12]: print(issue)
    if issues: raise SystemExit(1)
    if args.package:
        # Exclusive creation: never overwrite an earlier deliverable.
        with zipfile.ZipFile(args.package,'x',zipfile.ZIP_DEFLATED) as archive:
            for path in args.dashboard.rglob('*'):
                if path.is_file():
                    archive.write(path, 'dashboard/'+path.relative_to(args.dashboard).as_posix())
            archive.write(args.checkpoint,'checkpoint/best.pth')
            archive.writestr('VERIFICACION.json',json.dumps(report,indent=2))
        print('Paquete creado:',args.package)

if __name__=='__main__': main()
