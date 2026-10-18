import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
docs = json.loads((ROOT / 'dashboard/data/documentos.json').read_text(encoding='utf-8'))

def md_to_html(md):
    lines = md.replace('\r\n', '\n').split('\n')
    html, in_table, in_list = '', False, False
    def fmt(t):
        t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
        t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
        return t
    for line in lines:
        s = line.strip()
        if s.startswith('|') and s.endswith('|'):
            if '---' in s:
                continue
            cells = [fmt(c.strip()) for c in s[1:-1].split('|')]
            if not in_table:
                if in_list:
                    html += '</ul>'
                    in_list = False
                html += '<table><thead><tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr></thead><tbody>'
                in_table = True
            else:
                html += '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
            continue
        elif in_table:
            html += '</tbody></table>'
            in_table = False
        if line.startswith('### '):
            if in_list:
                html += '</ul>'
                in_list = False
            html += f'<h3>{fmt(line[4:])}</h3>'
        elif line.startswith('## '):
            if in_list:
                html += '</ul>'
                in_list = False
            html += f'<h2>{fmt(line[3:])}</h2>'
        elif line.startswith('# '):
            if in_list:
                html += '</ul>'
                in_list = False
            html += f'<h1>{fmt(line[2:])}</h1>'
        elif s.startswith('- ') or s.startswith('* '):
            if not in_list:
                html += '<ul>'
                in_list = True
            html += f'<li>{fmt(s[2:])}</li>'
        elif s == '':
            if in_list:
                html += '</ul>'
                in_list = False
        else:
            if in_list:
                html += '</ul>'
                in_list = False
            html += f'<p>{fmt(line)}</p>'
    if in_table:
        html += '</tbody></table>'
    if in_list:
        html += '</ul>'
    return html

cards_html = ''
for idx, d in enumerate(docs):
    cards_html += f'<section class="card" id="doc-{idx}"><div class="badge-doc">{d["title"]}</div>{md_to_html(d["content"])}</section>'

nav_html = ''.join(f'<a href="#doc-{idx}">{d["title"]}</a>' for idx, d in enumerate(docs))
full_html = f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><title>PENGWIN · Model Cards e Informes Visuales</title>
<style>
body {{ margin:0; font-family: "Segoe UI", system-ui, sans-serif; background:#0b1524; color:#e2e8f0; display:grid; grid-template-columns:340px 1fr; height:100vh; overflow:hidden; }}
aside {{ background:#0f1d30; border-right:1px solid #233854; padding:20px; overflow-y:auto; display:flex; flex-direction:column; gap:8px; }}
aside h2 {{ color:#38bdf8; font-size:1.15rem; margin:0 0 10px; }}
aside a {{ color:#cbd5e1; text-decoration:none; padding:10px 12px; border-radius:8px; background:#14263d; border:1px solid #263e5c; font-size:0.9rem; transition:0.15s; }}
aside a:hover {{ background:#1e3a5f; color:#fff; border-color:#38bdf8; }}
main {{ padding:32px 48px; overflow-y:auto; scroll-behavior:smooth; }}
.card {{ background:#111f33; border:1px solid #284263; border-radius:14px; padding:28px 36px; margin-bottom:36px; box-shadow:0 8px 24px rgba(0,0,0,0.35); }}
.badge-doc {{ display:inline-block; background:#1e3a5f; color:#7dd3fc; border:1px solid #38bdf8; border-radius:999px; padding:4px 14px; font-weight:600; font-size:0.85rem; margin-bottom:12px; }}
h1, h2, h3 {{ color:#38bdf8; border-bottom:1px solid #1f3552; padding-bottom:6px; }}
table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:0.92rem; }}
th, td {{ border:1px solid #2a4365; padding:10px 12px; text-align:left; }}
th {{ background:#172a42; color:#7dd3fc; }}
tr:nth-child(even) {{ background:rgba(255,255,255,0.03); }}
code {{ background:#1e293b; color:#f472b6; padding:2px 6px; border-radius:4px; }}
</style></head><body>
<aside><h2>🗂️ Model Cards e Informes</h2>{nav_html}</aside>
<main>{cards_html}</main></body></html>'''

(ROOT / 'salidas/MODEL_CARDS_VISUAL.html').write_text(full_html, encoding='utf-8')
(ROOT / 'doc/MODEL_CARDS_VISUAL.html').write_text(full_html, encoding='utf-8')
print('MODEL_CARDS_VISUAL.html generado exitosamente')
