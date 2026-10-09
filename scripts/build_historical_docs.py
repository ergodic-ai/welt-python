"""Retain SDK 0.6 routes using its immutable public notice-corrected docs revision."""
import argparse
import json
from pathlib import Path
import re
import subprocess

REVISION = '1cea7d9998bf433bd4404add75ed9e2e35b2de87'
TAG = 'v0.6.0'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument("--version", choices=("0.6.0", "0.7.0", "0.7.1"), default="0.6.0")
    args = parser.parse_args()
    revision = {"0.6.0": REVISION, "0.7.0": "0d7719f410061ac41f9fdc0428cfe44148999489", "0.7.1": "a14e880ea17fa4114bacf7b269283c4db8327678"}[args.version]
    tag = "v" + args.version
    source = args.source.resolve()
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if actual != revision:
        raise SystemExit('Historical documentation must use the pinned notice-corrected public revision.')
    # Only source links change in this disposable historical checkout. The SDK
    # contract stays 0.6, retaining the corrected archive/notice guide from 1cea.
    for path in (source/'docs').glob('*.md'):
        text = path.read_text()
        text = re.sub(r'(https://github.com/ergodic-ai/welt-python/(?:blob|tree)/)(?:main|feat/causal-client-preview)/', r'\g<1>'+tag+'/', text)
        text = text.replace('https://raw.githubusercontent.com/ergodic-ai/welt-python/main/', 'https://raw.githubusercontent.com/ergodic-ai/welt-python/'+tag+'/')
        path.write_text(text)
    subprocess.run([str(source/'.venv/bin/python'), str(source/'scripts/build_docs.py'), '--output', str(args.output.resolve())], cwd=source, check=True)
    if args.version == "0.7.1":
        # User requested presentation corrections on this exact old route.
        # Only landing HTML/CSS/JS change; its runtime, guides and release stay pinned.
        import hashlib
        import shutil
        if __package__:
            from .build_docs import home_body, shell
        else:
            from build_docs import home_body, shell
        out = args.output.resolve()/tag
        start = (source/'docs/start.md').read_text()
        (out/'index.html').write_text(shell(home_body(start,sdk_version='0.7.1',browser_connect=False),
            'index','From a DataFrame to a prediction','0.7.1',home=True))
        current = Path(__file__).resolve().parents[1]
        for name in ('docs.css','docs.js'):
            shutil.copy2(current/'docs/assets'/name,out/'assets'/name)
    manifest = args.output.resolve()/tag/'manifest.json'
    value = json.loads(manifest.read_text())
    value.update(docs_source_revision=revision, sdk_source_tag=tag, historical_source_link_fix=f'Example links pinned to matching {tag}; original routes/contracts retained.')
    if args.version == "0.7.1":
        value['homepage_presentation_override'] = dict(source_tag='v0.8.0',
            scope='API-072 title/numbered steps/visible train DataFrame; no0.8 API advertised',
            html_sha256=hashlib.sha256((out/'index.html').read_bytes()).hexdigest(),
            assets={name:hashlib.sha256((out/'assets'/name).read_bytes()).hexdigest() for name in ('docs.css','docs.js')})
    manifest.write_text(json.dumps(value, indent=2)+'\n')


if __name__ == '__main__':
    main()
