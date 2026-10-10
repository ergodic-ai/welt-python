"""Regression checks for immutable SDK/docs-presentation boundaries."""
import os
from unittest import TestCase, main
from unittest.mock import patch
from scripts import verify_docs_presentation as guard

class Boundaries(TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,DOCS_PRESENTATION_REF='a'*40,RELEASE_TAG='v0.8.0')
        self.env.start();self.addCleanup(self.env.stop)
    def run_guard(self, changes):
        def git(*args):
            if args==('rev-parse','HEAD'):return 'a'*40
            if args==('rev-parse','refs/tags/v0.8.0'):return guard.BASELINE
            if args==('diff','--name-status',guard.BASELINE,'a'*40):return changes
            if args==('status','--porcelain'):return ''
            raise AssertionError(args)
        with patch.object(guard,'git',side_effect=git),patch.object(guard.subprocess,'run'),patch.object(guard.urllib.request,'urlopen') as network:
            with self.assertRaises(SystemExit):guard.main()
            network.assert_not_called()
    def test_moving_ref_rejected_before_any_git_or_network(self):
        with patch.dict(os.environ,DOCS_PRESENTATION_REF='main'),patch.object(guard,'git') as git:
            with self.assertRaises(SystemExit):guard.main()
            git.assert_not_called()
    def test_other_release_rejected(self):
        with patch.dict(os.environ,RELEASE_TAG='v0.7.1'):
            self.run_guard('M\tdocs/assets/docs.css')
    def test_runtime_edit_rejected(self):self.run_guard('M\tsrc/welt/client.py')
    def test_dependency_edit_rejected(self):self.run_guard('M\tuv.lock')
    def test_notebook_edit_rejected(self):self.run_guard('M\texamples/notebooks/first_prediction.ipynb')
    def test_delete_rejected(self):self.run_guard('D\tdocs/assets/docs.css')
    def test_rename_rejected(self):self.run_guard('R100\tdocs/assets/docs.css\tdocs/assets/other.css')

if __name__=='__main__':main()
