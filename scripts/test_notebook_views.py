"""Notebook destinations, source fidelity and current learning routes."""
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tomllib
import tempfile
from unittest import TestCase
from unittest.mock import patch

from scripts import build_docs, verify_docs_presentation


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links = []
        self.copy = []
        self.ids = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'a':
            self.links.append(values)
        if 'data-copy' in values:
            self.copy.append(values['data-copy'])
        if 'id' in values:
            self.ids.append(values['id'])


class NotebookViews(TestCase):
    def test_all_actions_match_the_downloaded_and_rendered_release_source(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(sys, 'argv', ['build_docs', '--output', folder]):
                build_docs.main()
            version = tomllib.loads((build_docs.ROOT/'pyproject.toml').read_text())['project']['version']
            site = Path(folder) / ('v' + version)
            index = Page((site / 'notebooks.html').read_text())
            self.assertNotIn('{{actions:', (site / 'notebooks.html').read_text())
            notebooks = sorted((build_docs.ROOT / 'examples/notebooks').glob('*.ipynb'))
            self.assertEqual(len(notebooks), 6)
            for source in notebooks:
                name = source.stem
                revision = build_docs.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=build_docs.ROOT, text=True).strip()
                colab = f'https://colab.research.google.com/github/ergodic-ai/welt-python/blob/{revision}/examples/notebooks/{name}.ipynb'
                self.assertEqual(sum(link.get('href') == colab for link in index.links), 1)
                download = next(link for link in index.links if link.get('href') == f'downloads/{name}.ipynb')
                self.assertEqual(download['download'], source.name)
                self.assertEqual((site / download['href']).read_bytes(), source.read_bytes())
                self.assertTrue(any(link.get('href') == f'notebook-{name}.html' for link in index.links))
                view_text = (site / f'notebook-{name}.html').read_text()
                view = Page(view_text)
                notebook = json.loads(source.read_text())
                self.assertEqual(view.copy, [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code'])
                self.assertIn('Static notebook preview', view_text)
                self.assertIn('notebooks.html#run-in-colab', view_text)
                self.assertIn(colab, view_text)
                self.assertEqual(len(view.ids), len(set(view.ids)))
                # Every markdown cell precedes the same next code cell as in the source.
                position = 0
                for cell in notebook['cells']:
                    text = ''.join(cell['source'])
                    rendered = build_docs.render_markdown(text)[0] if cell['cell_type'] == 'markdown' else 'data-copy="' + build_docs.html.escape(text, quote=True) + '"'
                    found = view_text.find(rendered, position)
                    self.assertGreaterEqual(found, position)
                    position = found + len(rendered)
            search = json.loads((site / 'search-index.json').read_text())
            manifest = json.loads((site / 'manifest.json').read_text())
            for name in manifest['notebook_views']:
                self.assertTrue(any(record['href'] == name + '.html' for record in search))
                self.assertIn(name + '.html', (Path(folder) / 'sitemap.xml').read_text())

    def test_html_and_fences_in_code_stay_literal_and_are_never_executed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sentinel = root / 'executed'
            code = f'from pathlib import Path\nPath({str(sentinel)!r}).touch()\ntext = "``` <script>alert(1)</script> & \\\""\n'
            source = root / 'literal.ipynb'
            source.write_text(json.dumps({'cells': [
                {'cell_type': 'markdown', 'source': ['# Literal example\n', '\nRead before running.']},
                {'cell_type': 'code', 'source': code.splitlines(keepends=True), 'outputs': [], 'execution_count': None}]}))
            body, title = build_docs.notebook_view(source, '0.8.0')
            self.assertEqual(title, 'Literal example')
            self.assertEqual(Page(body).copy, [code])
            self.assertNotIn('<script>', body)
            self.assertIn('&lt;script&gt;', body)
            self.assertFalse(sentinel.exists())

    def test_saved_outputs_or_execution_counts_refuse_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'saved.ipynb'
            for outputs, count in [([{'output_type': 'stream', 'text': 'private value'}], None), ([], 1)]:
                source.write_text(json.dumps({'cells': [
                    {'cell_type': 'markdown', 'source': '# Saved'},
                    {'cell_type': 'code', 'source': 'print("example")', 'outputs': outputs, 'execution_count': count}]}))
                with self.assertRaisesRegex(SystemExit, 'require cleared outputs'):
                    build_docs.notebook_view(source, '0.8.0')

    def test_presentation_scope_allows_index_and_tests_but_not_notebook_edits(self):
        self.assertIn('docs/notebooks.md', verify_docs_presentation.LEARNING_ALLOWLIST)
        self.assertIn('scripts/test_notebook_views.py', verify_docs_presentation.LEARNING_ALLOWLIST)
        self.assertNotIn('examples/notebooks/classification.ipynb', verify_docs_presentation.ALLOWLIST)

    def test_old_learning_routes_reach_current_docs_and_downloads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root/'v0.8.0'
            (old/'downloads').mkdir(parents=True)
            (old/'notebooks.html').write_text('older learning page')
            (old/'notebook-classification.html').write_text('older example')
            (old/'reference.html').write_text('older reference')
            with patch.object(sys, 'argv', ['build_docs', '--output', folder]):
                build_docs.main()
            for page in ('notebooks.html', 'notebook-classification.html', 'reference.html'):
                redirected = (old/page).read_text()
                self.assertIn('href="../' + page + '"', redirected)
                self.assertIn('destination.hash = window.location.hash', redirected)
                self.assertIn('destination.search = window.location.search', redirected)
                self.assertNotIn('http-equiv="refresh"', redirected)
            self.assertEqual((root/'downloads/classification.ipynb').read_bytes(),
                             (old/'downloads/classification.ipynb').read_bytes())
            self.assertTrue((root/'research-datasets.html').exists())
            self.assertTrue((root/'assets/docs.js').exists())
