"""Build versioned static SDK guides from owned Markdown and installed SDK signatures."""
from pathlib import Path
import argparse
import html
import inspect
from importlib.metadata import version as installed_version
import json
import re
import tomllib

import markdown
import welt

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='site');args=parser.parse_args()
    version=tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['version']
    if installed_version('welt-client') != version or getattr(welt,'__version__',None) != version:
        raise SystemExit('Installed SDK version must match the documentation source version.')
    required=('Client','Job','Classifier','Regressor','AsyncClient','AsyncJob')
    if any(not hasattr(welt,name) for name in required):
        raise SystemExit('Installed SDK is missing a required documented public symbol.')
    out=Path(args.output)/f'v{version}';out.mkdir(parents=True,exist_ok=True)
    pages={p.stem:p.read_text() for p in sorted((ROOT/'docs').glob('*.md'))}
    references=['# Installed SDK reference','Signatures come from the installed package; availability depends on the service capability catalogue.']
    for name in required:
        obj=getattr(welt,name,None)
        references.append(f'## {name}\n\n```python\n{name}{inspect.signature(obj)}\n```')
        for key,method in inspect.getmembers(obj,inspect.isroutine):
            if not key.startswith('_') and method.__module__.startswith('welt'):
                references.append(f'```python\n{name}.{key}{inspect.signature(method)}\n```')
    pages['reference']='\n\n'.join(references)
    navigation=' '.join(f'<a href="{html.escape(name)}.html">{html.escape(name.replace("-"," "))}</a>' for name in pages)
    css='body{font:17px/1.6 system-ui;max-width:960px;margin:2rem auto;padding:0 1rem;color:#172536}nav a{margin-right:1rem}pre{background:#f0f4f8;padding:1rem;overflow:auto}code{font-size:.9em}table{border-collapse:collapse}td,th{border:1px solid #ccd6e0;padding:.4rem}a{color:#075bb0}'
    for name,text in pages.items():
        text=re.sub(r'\]\(([^():#]+)\.md\)',r'](\1.html)',text)
        body=markdown.markdown(text,extensions=['fenced_code','tables'])
        (out/f'{name}.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Welt SDK {html.escape(version)} — {html.escape(name)}</title><style>{css}</style><body><header><b>Welt SDK {html.escape(version)}</b></header><nav>{navigation}</nav><main>{body}</main></body></html>')
    (out/'index.html').write_text((out/'usage.html').read_text())
    base=out.parent;base.mkdir(parents=True,exist_ok=True)
    (base/'index.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Welt SDK documentation</title><body><a href="v{html.escape(version)}/">Welt SDK {html.escape(version)} documentation</a></body></html>')
    (base/'.nojekyll').write_text('')
    manifest=dict(sdk_version=version,pages=sorted(pages),supported_notebooks=['classification','artifacts-and-usage','async-jobs'],scope='incremental; full-model/SDK closure gates pending')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Built {len(pages)} documentation pages for SDK {version}; no credentials or outputs included')


if __name__=='__main__':main()
