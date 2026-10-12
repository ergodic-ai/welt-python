"""Guard reviewed documentation updates against immutable SDK release bytes."""
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

RELEASES = {
    '0.8.0': (BASELINE, WHEEL_SHA),
    '0.10.0': ('f41f37e1b69f178969707f8b31e58b85e71fb158',
               'b079af10218b5c37f0ebcf9a11c906b3703e88c53b462050ac165d766689101f'),
}
LEARNING_ALLOWLIST = ALLOWLIST | {
    'docs/' + name + '.md' for name in (
        'start', 'notebooks', 'research-datasets', 'onboarding', 'usage', 'model-runs', 'migration')
} | {
    'examples/first_prediction.py', 'examples/first_request.py',
    'examples/notebooks/README.md', 'scripts/check_notebooks.py',
    'scripts/notebook_fixture.py', 'scripts/test_notebook_views.py',
    'tests/test_first_request.py',
} | {
    'examples/notebooks/' + name + '.ipynb' for name in (
        'classification', 'regression', 'async-jobs', 'artifacts-and-usage',
        'research-datasets', 'causal-discovery')
}

def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()

def main():
    ref = os.environ['DOCS_PRESENTATION_REF']
    release = os.environ['RELEASE_TAG']
    version = release.removeprefix('v')
    if not re.fullmatch('[0-9a-f]{40}', ref) or release != 'v' + version or version not in RELEASES:
        raise SystemExit('Presentation mode requires an exact reviewed commit and supported immutable SDK tag.')
    baseline, wheel_sha = RELEASES[version]
    if version == '0.8.0':baseline, wheel_sha = BASELINE, WHEEL_SHA
    allowlist = LEARNING_ALLOWLIST if version == '0.10.0' else ALLOWLIST
    if git('rev-parse', 'HEAD') != ref or git('rev-parse', f'refs/tags/{release}^{{commit}}') != baseline:
        raise SystemExit('Presentation checkout or immutable SDK tag differs.')
    subprocess.run(['git', 'merge-base', '--is-ancestor', baseline, ref], check=True)
    changes = git('diff', '--name-status', baseline, ref).splitlines()
    files = []
    for line in changes:
        fields = line.split('\t')
        if len(fields) != 2:
            raise SystemExit('Renames are outside the documentation presentation scope.')
        status, name = fields
        if status not in ('A', 'M') or name not in allowlist:
            raise SystemExit('Presentation change exceeds reviewed documentation allowlist.')
        files.append(name)
    if not files or git('status', '--porcelain'):
        raise SystemExit('Presentation checkout must be clean and contain documentation changes.')
    request = urllib.request.Request(
        f'https://pypi.org/pypi/welt-client/{version}/json',
        headers={'User-Agent': 'Welt-docs-presentation'})
    with urllib.request.urlopen(request, timeout=30) as response:
        published = json.load(response)
    entries = [entry for entry in published['urls'] if entry['filename'] == f'welt_client-{version}-py3-none-any.whl']
    if published['info']['version'] != version or len(entries) != 1 or entries[0]['digests']['sha256'] != wheel_sha:
        raise SystemExit('Published immutable SDK wheel differs.')
    url = entries[0]['url']
    if not url.startswith('https://files.pythonhosted.org/'):
        raise SystemExit('Unexpected published wheel host.')
    with urllib.request.urlopen(url, timeout=30) as response:
        wheel = response.read(4 * 1024 * 1024)
    if hashlib.sha256(wheel).hexdigest() != wheel_sha:
        raise SystemExit('Published wheel download differs.')
    Path('/tmp/welt-docs-presentation-wheel.whl').write_bytes(wheel)
    proof = {'presentation_commit': ref, 'sdk_source_commit': baseline,
             'sdk_version': version, 'published_wheel_sha256': wheel_sha,
             'runtime_dependencies_unchanged': True,
             'teaching_sources_updated': version == '0.10.0',
             'changed_files': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}}
    Path('/tmp/welt-docs-presentation-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof))

if __name__ == '__main__':
    main()
