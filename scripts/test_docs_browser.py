"""API-122: keyboard, theme, phone-width and component checks in a real browser.

Runs when Chrome and Node 22+ are available (GitHub's ubuntu runners have both);
otherwise skipped. Contrast checks below always run.
"""
from functools import partial
import http.server
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from unittest.mock import patch

import pytest

from scripts import build_docs

ASSETS = build_docs.ROOT / 'docs/assets'


def chrome():
    candidates = [os.environ.get('CHROME'), shutil.which('google-chrome'), shutil.which('chromium'), shutil.which('chromium-browser'),
                  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome']
    return next((path for path in candidates if path and Path(path).exists()), None)


def node_ok():
    node = shutil.which('node')
    if not node:
        return False
    major = int(re.match(r'v(\d+)', subprocess.run([node, '--version'], capture_output=True, text=True).stdout)[1])
    return major >= 22


@pytest.mark.skipif(not (chrome() and node_ok()), reason='Chrome and Node 22+ are required for browser checks')
def test_rendered_site_in_a_real_browser():
    with tempfile.TemporaryDirectory() as folder:
        with patch.object(sys, 'argv', ['build_docs', '--output', folder]):
            build_docs.main()
        version = json.loads(next(Path(folder).glob('v*/manifest.json')).read_text())['sdk_version']
        handler = partial(http.server.SimpleHTTPRequestHandler, directory=folder)
        handler.log_message = lambda *args: None
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            run = subprocess.run(['node', str(build_docs.ROOT / 'scripts/docs_browser_check.mjs'), chrome(),
                                  f'http://127.0.0.1:{server.server_port}/v{version}', str(Path(folder) / 'profile')],
                                 capture_output=True, text=True, timeout=240)
        finally:
            server.shutdown()
    results = json.loads(run.stdout.strip().splitlines()[-1])
    assert results.pop('errors') == []
    assert results.pop('overflow') == {}
    assert results.pop('searchResults') > 0
    assert all(value is True for value in results.values()), results


def luminance(colour):
    channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(foreground, background):
    high, low = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def palette(theme):
    """Resolved hex tokens for one theme: console tokens, then docs-only overrides."""
    values = {}
    sources = [(ASSETS / 'tokens.css').read_text(), (ASSETS / 'docs.css').read_text()]
    for css in sources:
        blocks = re.findall(r'(:root(?:\[data-theme="dark"\])?)\s*\{([^{}]*)\}', css)
        for selector, body in blocks:
            if (selector == ':root') or (theme == 'dark' and selector == ':root[data-theme="dark"]'):
                values.update(re.findall(r'(--[\w-]+):\s*(#[0-9a-fA-F]{6})\b', body))
    return values


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_text_and_code_colours_meet_wcag_aa(theme):
    tokens = palette(theme)
    pairs = [('--foreground', '--background'), ('--muted-foreground', '--background'), ('--muted-foreground', '--sidebar'),
             ('--foreground', '--secondary'), ('--code-comment', '--code-background'), ('--code-keyword', '--code-background'),
             ('--code-string', '--code-background'), ('--code-function', '--code-background'), ('--code-number', '--code-background'),
             ('--info', '--background'), ('--primary-foreground', '--primary')]
    for foreground, background in pairs:
        ratio = contrast(tokens[foreground], tokens[background])
        assert ratio >= 4.5, (theme, foreground, background, round(ratio, 2))
