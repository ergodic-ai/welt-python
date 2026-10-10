"""Regression checks for immutable SDK/docs-presentation boundaries."""
import os
from pathlib import Path
import subprocess
import tempfile
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
            if args==('rev-parse','refs/tags/v0.8.0^{commit}'):return guard.BASELINE
            if args==('diff','--name-status',guard.BASELINE,'a'*40):return changes
            if args==('status','--porcelain'):return ''
            raise AssertionError(args)
        with patch.object(guard,'git',side_effect=git),patch.object(guard.subprocess,'run'),patch.object(guard.urllib.request,'urlopen') as network:
            with self.assertRaises(SystemExit):guard.main()
            network.assert_not_called()
    def test_annotated_tag_is_compared_as_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def command(*args):
                return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
            command('init');command('config','user.name','Guard fixture');command('config','user.email','guard@example.invalid')
            file=root/'docs/assets/docs.css';file.parent.mkdir(parents=True);file.write_text('baseline')
            command('add','.');command('commit','-m','baseline');baseline=command('rev-parse','HEAD')
            command('tag','-a','v0.8.0','-m','annotated SDK fixture')
            self.assertNotEqual(command('rev-parse','refs/tags/v0.8.0'),baseline)
            file.write_text('presentation');command('add','.');command('commit','-m','presentation');candidate=command('rev-parse','HEAD')
            previous=os.getcwd();os.chdir(root)
            try:
                with patch.object(guard,'BASELINE',baseline),patch.dict(os.environ,DOCS_PRESENTATION_REF=candidate),patch.object(guard.urllib.request,'urlopen',side_effect=RuntimeError('reached published-wheel verification')) as network:
                    with self.assertRaisesRegex(RuntimeError,'reached published-wheel verification'):guard.main()
                    network.assert_called_once()
            finally:os.chdir(previous)
    def test_wrong_tag_commit_rejected_before_network(self):
        with patch.object(guard,'git',side_effect=['a'*40,'b'*40]),patch.object(guard.urllib.request,'urlopen') as network:
            with self.assertRaisesRegex(SystemExit,'immutable SDK tag differs'):guard.main()
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
