"""Prevent docs from falsely publishing a mismatched/incomplete installed SDK reference."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_docs


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


if __name__=='__main__':unittest.main()
