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
    parser.add_argument("--version", choices=("0.6.0", "0.7.0"), default="0.6.0")
    args = parser.parse_args()
    revision = REVISION if args.version == "0.6.0" else "0d7719f410061ac41f9fdc0428cfe44148999489"
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
    manifest = args.output.resolve()/tag/'manifest.json'
    value = json.loads(manifest.read_text())
    value.update(docs_source_revision=revision, sdk_source_tag=tag, historical_source_link_fix=f'Example links pinned to matching {tag}; original routes/contracts retained.')
    manifest.write_text(json.dumps(value, indent=2)+'\n')


if __name__ == '__main__':
    main()
