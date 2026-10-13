"""Prevent docs from falsely publishing a mismatched/incomplete installed SDK reference."""
import sys
import importlib.util

import numpy as np
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_docs, check_notebooks


class ReferenceGates(unittest.TestCase):
    def test_stale_installed_package_cannot_publish_current_reference(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(sys,'argv',['build_docs','--output',folder]), patch.object(build_docs,'installed_version',return_value='0.0.0'):
            with self.assertRaisesRegex(SystemExit,'version must match'):
                build_docs.main()
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_missing_async_interface_cannot_silently_disappear(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(sys,'argv',['build_docs','--output',folder]):
            with patch.object(build_docs.welt,'AsyncClient'):
                original=build_docs.welt.AsyncClient
                del build_docs.welt.AsyncClient
                try:
                    with self.assertRaisesRegex(SystemExit,'required documented public symbol'):
                        build_docs.main()
                finally:build_docs.welt.AsyncClient=original
            self.assertEqual(list(Path(folder).iterdir()),[])


class NotebookSelection(unittest.TestCase):
    def test_unknown_selection_rejects_before_kernel_execution(self):
        with patch.object(sys,'argv',['check_notebooks','--notebook','not-available']), patch.object(check_notebooks,'NotebookClient') as execute:
            with self.assertRaises(SystemExit):check_notebooks.main()
            execute.assert_not_called()


class TeachingSources(unittest.TestCase):
    def test_notebooks_are_standalone_real_data_and_scrubbed(self):
        import json
        root=Path(__file__).resolve().parents[1]
        for path in (root/'examples/notebooks').glob('*.ipynb'):
            notebook=json.loads(path.read_text())
            code="\n".join("".join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code')
            first = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code')
            self.assertEqual(first['source'], '!pip install "welt-client[parquet]"')
            self.assertNotIn('__target__#0', code)
            self.assertNotIn('version=', code)
            self.assertNotIn('license_notices', code)
            self.assertIn('research_dataset',code)
            self.assertIn('download_research_dataset',code)
            self.assertIn('pd.read_parquet',code)
            expected_target = 'Residuary.resistance' if path.stem == 'regression' else 'target'
            self.assertIn('target = "' + expected_target + '"', code)
            self.assertIn('data.head()',code)
            for forbidden in ('estimator_transport','from support','MockTransport','make_classification','np.random'):
                self.assertNotIn(forbidden,code)
            if path.stem!='causal-discovery':
                fit = next(call for call in ('model.fit(', 'model.submit_fit(', 'evaluation.fit(') if call in code)
                self.assertLess(code.index('train_test_split('),code.index(fit))
                if path.stem=='evaluation':  # Cross-validation sees only the training rows.
                    self.assertIn('client.evaluate(train, target',code)
                self.assertIn('test.drop(columns=target)',code)
                self.assertTrue('accuracy_score' in code or 'mean_absolute_error' in code)
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    self.assertEqual(cell['outputs'],[]);self.assertIsNone(cell['execution_count'])

    def test_fixture_refuses_unverified_canonical_bytes(self):
        from scripts.notebook_fixture import ResearchFixture
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'iris.json').write_text('{}');(root/'iris.parquet').write_bytes(b'not research')
            with self.assertRaises(AssertionError):ResearchFixture(root)


class SiteContracts(unittest.TestCase):
    def build(self, folder):
        with patch.object(sys, 'argv', ['build_docs', '--output', folder]):
            build_docs.main()

    def test_versioned_links_search_anchors_and_historical_preservation(self):
        from html.parser import HTMLParser
        from urllib.parse import urlsplit, unquote
        import json
        class Document(HTMLParser):
            def __init__(self, text):
                super().__init__(); self.ids=set(); self.links=[]; self.complete=[]; self.feed(text)
            def handle_starttag(self, tag, attrs):
                values=dict(attrs)
                if 'id' in values:self.ids.add(values['id'])
                if tag=='a' and 'href' in values:self.links.append(values['href'])
                if 'data-copy' in values:self.complete.append(values['data-copy'])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old=root/'v0.6.0';old.mkdir();(old/'index.html').write_text('historical sentinel')
            self.build(folder)
            self.assertEqual((old/'index.html').read_text(),'historical sentinel')
            current=root/('v'+tomllib.loads((build_docs.ROOT/'pyproject.toml').read_text())['project']['version'])
            documents={p.resolve():Document(p.read_text()) for p in root.rglob('*.html') if p.parent!=old}
            for path,doc in documents.items():
                for href in doc.links:
                    parts=urlsplit(href)
                    if parts.scheme or parts.netloc:continue
                    target=(path.parent/unquote(parts.path)).resolve() if parts.path else path
                    self.assertTrue(target.exists(),f'{path.name}: {href}')
                    if parts.fragment and target in documents:
                        self.assertIn(unquote(parts.fragment),documents[target].ids,f'{path.name}: {href}')
            index=json.loads((current/'search-index.json').read_text())
            self.assertTrue(any('JobTimeoutError' in item['text'] for item in index))
            for item in index:
                parts=urlsplit(item['href']);target=(current/parts.path).resolve()
                self.assertIn(target,documents)
                if parts.fragment:self.assertIn(parts.fragment,documents[target].ids)
            self.assertEqual(len(documents[(root/'index.html').resolve()].complete),0)

    def test_release_versions_tolerate_prereleases_and_list_every_older_historical_site(self):
        self.assertEqual(build_docs.release_tuple('0.9.0rc1'),(0,9,0))
        self.assertEqual(build_docs.release_tuple('1.0.0.dev2+local'),(1,0,0))
        self.assertTrue(build_docs.browser_connect_release('0.9.0rc1'))
        self.assertTrue(build_docs.browser_connect_release('0.10.0'))
        self.assertFalse(build_docs.browser_connect_release('0.7.1'))
        self.assertEqual(build_docs.older_versions('0.9.0'),['0.8.0','0.7.1','0.7.0','0.6.0'])
        self.assertEqual(build_docs.older_versions('0.8.0'),['0.7.1','0.7.0','0.6.0'])
        self.assertEqual(build_docs.older_versions('0.7.1'),['0.7.0','0.6.0'])
        with self.assertRaises(ValueError):build_docs.release_tuple('latest')
        page=build_docs.shell('<p>x</p>','start','Start','0.9.0')
        self.assertIn('<option value="../v0.8.0/">v0.8.0</option>',page)
        self.assertNotIn('v0.9.0/">',page)

    def test_historical_v080_folder_is_in_sitemap_and_manifest(self):
        import json
        version=tomllib.loads((build_docs.ROOT/'pyproject.toml').read_text())['project']['version']
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for old in ('0.6.0','0.8.0'):
                (root/f'v{old}').mkdir();(root/f'v{old}'/'index.html').write_text('historical sentinel')
            self.build(folder)
            sitemap=(root/'sitemap.xml').read_text()
            manifest=json.loads((root/f'v{version}'/'manifest.json').read_text())
            expected=[old for old in ('0.6.0','0.8.0') if build_docs.release_tuple(old)<build_docs.release_tuple(version)]
            self.assertEqual(manifest['historical_versions'],expected)
            for old in expected:self.assertIn(f'/v{old}/</loc>',sitemap)
            self.assertNotIn('/v0.7.0/',sitemap)

    def test_landing_copy_loads_data_visibly_before_modelling(self):
        from scripts.test_homepage_presentation import NotebookPage
        with tempfile.TemporaryDirectory() as folder:
            self.build(folder)
            body=(Path(folder)/'index.html').read_text()
            parsed=NotebookPage();parsed.feed(body)
            self.assertNotIn('data-copy=',body)
            self.assertIn('Load Iris below',body)
            program=next(value for value in parsed.code if 'train, test = train_test_split' in value)
            compile(program,'visible copied cell','exec')
            self.assertIn('model.fit(X_train, y_train)',program)
            self.assertIn('model.predict(X_test)',program)
            self.assertIn('classification_report',program)
            self.assertNotIn('download_research_dataset',program)
            self.assertTrue(any('load_research_dataset' in code and 'data.head()' in code for code in parsed.code))

    def test_mutable_source_link_refuses_before_writing_site(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(sys,'argv',['build_docs','--output',folder]), patch.object(build_docs,'reference',return_value='# Reference\n\n[bad](https://github.com/ergodic-ai/welt-python/blob/main/examples/x.py)'):
            with self.assertRaisesRegex(SystemExit,'mutable SDK source'):build_docs.main()
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_historical_revision_mismatch_leaves_source_untouched(self):
        from scripts import build_historical_docs
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'docs').mkdir();notice=root/'docs/research-datasets.md';notice.write_text('retained licence notice')
            with patch.object(sys,'argv',['historical','--source',folder,'--output',folder+'/site']), patch.object(build_historical_docs.subprocess,'check_output',return_value='wrong-source\n'), patch.object(build_historical_docs.subprocess,'run') as execute:
                with self.assertRaisesRegex(SystemExit,'pinned notice-corrected'):build_historical_docs.main()
                execute.assert_not_called()
                self.assertEqual(notice.read_text(),'retained licence notice')

    def test_unannotated_public_method_cannot_publish_signature_only(self):
        def missing(self):pass
        missing.__module__='welt.client'
        with patch.object(build_docs.welt.Client,'unannotated',missing,create=True):
            with self.assertRaisesRegex(SystemExit,'Missing annotated method contract'):build_docs.reference()


if __name__=='__main__':unittest.main()


def test_forecast_reference_describes_zero_shot_context_and_typed_ownership():
    text = build_docs.reference()
    section = text.split('### Forecast.fit',1)[1].split('###',1)[0]
    assert 'No history upload, predictor or training job' in section
    assert 'new immutable predictor' not in section
    assert '### Forecast.predict_quantiles' in text
    assert '### AsyncForecastJob.result' in text
    assert 'ForecastJob owning its new client' in text
    assert 'Reconnected handles borrow' in text
    assert 'InvalidForecastResultError' in text
    guide = (build_docs.ROOT/'docs/forecasting.md').read_text()
    assert 'cannot reconstruct' in guide and 'not calibrated' in guide
    assert any(slug == 'forecasting' for _, group in build_docs.GUIDES for _, slug in group)
