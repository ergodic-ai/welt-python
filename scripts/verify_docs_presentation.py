"""Guard an explicitly reviewed docs-only presentation checkout against SDK0.8."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request

BASELINE = '49699b14ebc10bcb093c9cf208518dd5ea3a0c46'
WHEEL_SHA = '38fc95dc0db20567db9f28604bdf079f53bc0e9a03f7cd3b45b59d9457c3a34e'
ALLOWLIST = {
    '.github/workflows/sdk-docs.yml',
    'scripts/verify_docs_presentation.py',
    'scripts/build_docs.py',
    'scripts/test_homepage_presentation.py',
    'scripts/test_learning_materials.py',
    'scripts/test_docs_presentation_guard.py',
    'docs/assets/docs.css',
    'docs/assets/docs.js',
}

def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()

def main():
    ref = os.environ['DOCS_PRESENTATION_REF']
    if not re.fullmatch('[0-9a-f]{40}', ref) or os.environ['RELEASE_TAG'] != 'v0.8.0':
        raise SystemExit('Presentation mode requires an exact reviewed commit and SDK0.8 tag.')
    if git('rev-parse', 'HEAD') != ref or git('rev-parse', 'refs/tags/v0.8.0') != BASELINE:
        raise SystemExit('Presentation checkout or immutable SDK tag differs.')
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASELINE, ref], check=True)
    changes = git('diff', '--name-status', BASELINE, ref).splitlines()
    files = []
    for line in changes:
        fields = line.split('\t')
        if len(fields) != 2:
            raise SystemExit('Renames are outside the documentation presentation scope.')
        status, name = fields
        if status not in ('A', 'M') or name not in ALLOWLIST:
            raise SystemExit('Presentation change exceeds reviewed documentation allowlist.')
        files.append(name)
    if not files or git('status', '--porcelain'):
        raise SystemExit('Presentation checkout must be clean and contain documentation changes.')
    request = urllib.request.Request(
        'https://pypi.org/pypi/welt-client/0.8.0/json',
        headers={'User-Agent': 'Welt-docs-presentation'})
    with urllib.request.urlopen(request, timeout=30) as response:
        version = json.load(response)
    entries = [entry for entry in version['urls'] if entry['filename'] == 'welt_client-0.8.0-py3-none-any.whl']
    if version['info']['version'] != '0.8.0' or len(entries) != 1 or entries[0]['digests']['sha256'] != WHEEL_SHA:
        raise SystemExit('Published immutable SDK wheel differs.')
    url = entries[0]['url']
    if not url.startswith('https://files.pythonhosted.org/'):
        raise SystemExit('Unexpected published wheel host.')
    with urllib.request.urlopen(url, timeout=30) as response:
        wheel = response.read(4 * 1024 * 1024)
    if hashlib.sha256(wheel).hexdigest() != WHEEL_SHA:
        raise SystemExit('Published wheel download differs.')
    Path('/tmp/welt-docs-presentation-wheel.whl').write_bytes(wheel)
    proof = {'presentation_commit': ref, 'sdk_source_commit': BASELINE,
             'sdk_version': '0.8.0', 'published_wheel_sha256': WHEEL_SHA,
             'runtime_dependencies_guides_notebooks_unchanged': True,
             'changed_files': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}}
    Path('/tmp/welt-docs-presentation-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof))

if __name__ == '__main__':
    main()
