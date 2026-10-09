"""Prevent docs from falsely publishing a mismatched/incomplete installed SDK reference."""
import sys
import importlib.util

import numpy as np
import tempfile
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


class RegressionNotebookFixture(unittest.TestCase):
    @staticmethod
    def support():
        path=Path(__file__).resolve().parents[1]/'examples/notebooks/support.py'
        spec=importlib.util.spec_from_file_location('notebook_support',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_regression_points_reorder_reopen_and_task_identity(self):
        support=self.support()
        with patch.dict('os.environ',{'WELT_NOTEBOOK_MODE':'fixture','WELT_API_KEY':'real-key-must-not-be-read'}):
            with support.notebook_client() as client:
                columns,query,predictor=support.prepare_regression(client)
                result=client.predict(predictor['id'],columns=columns,rows=query.tolist())
                reordered=client.predict(predictor['id'],columns=columns[::-1],rows=query[:,::-1].tolist())
                self.assertEqual(predictor['task'],'regression')
                self.assertEqual(predictor['model_version'],'synthetic-regression-contract-v1')
                self.assertIsNone(result['classes']);self.assertIsNone(result['probabilities'])
                self.assertEqual(np.shape(result['predictions']),(32,))
                self.assertTrue(np.isfinite(result['predictions']).all())
                np.testing.assert_allclose(result['predictions'],reordered['predictions'])
                self.assertEqual(client.predictor(predictor['id'])['model_version'],predictor['model_version'])
                self.assertTrue(all(e['task']=='regression' for e in client.usage()))

    def test_missing_regression_profile_rejects_before_upload(self):
        support=self.support()
        from unittest.mock import Mock
        client=Mock();client.models.return_value=[dict(id='tabicl-v2',status='available',tasks=['classification'])]
        with self.assertRaisesRegex(RuntimeError,'Qualified regression worker unavailable'):
            support.prepare_regression(client)
        client.upload.assert_not_called();client.submit_fit.assert_not_called()

    def test_regression_rng_matches_frozen_planted_fixture(self):
        support=self.support();columns,train,target,query=support.regression_table()
        rng=np.random.default_rng(42);expected_train=rng.normal(size=(128,4));expected_query=rng.normal(size=(32,4))
        weights=rng.normal(size=4);expected_target=expected_train@weights+rng.normal(scale=0.05,size=128)
        self.assertEqual(len(columns),4)
        np.testing.assert_array_equal(train,expected_train);np.testing.assert_array_equal(query,expected_query)
        np.testing.assert_array_equal(target,expected_target)





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
            current=root/'v0.7.0'
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
            self.assertEqual(len(documents[(root/'index.html').resolve()].complete),2)

    def test_landing_complete_copy_examples_execute_without_live_requests(self):
        import contextlib, io
        from html.parser import HTMLParser
        class Examples(HTMLParser):
            def __init__(self,text):super().__init__();self.code=[];self.feed(text)
            def handle_starttag(self,tag,attrs):
                values=dict(attrs)
                if 'data-copy' in values:self.code.append(values['data-copy'])
        support=RegressionNotebookFixture.support()
        with tempfile.TemporaryDirectory() as folder:
            self.build(folder);examples=Examples((Path(folder)/'index.html').read_text()).code
            for value in examples:
                namespace={};output=io.StringIO()
                with support.estimator_transport(),contextlib.redirect_stdout(output):exec(value,namespace)
                self.assertEqual(namespace['predictions'].shape,(32,))
                self.assertEqual(output.getvalue().strip(),'ndarray (32,)')
                self.assertEqual(namespace['model'].n_features_in_,4)
                self.assertNotIn('label',namespace['model']._columns_)
                self.assertNotIn('target',namespace['model']._columns_)

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
