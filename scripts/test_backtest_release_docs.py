"""API-195: retain the released documentation while publishing backtesting."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from scripts import build_docs


def test_release_workflow_builds_old_sources_and_exact_published_wheels():
    workflow=(build_docs.ROOT/'.github/workflows/sdk-docs.yml').read_text()
    assert 'default: v0.16.0' in workflow
    for line in workflow.splitlines():
        if line.lstrip().startswith('if:') and "'v0.14.2'" in line:
            assert "github.event.release.tag_name == 'v0.15.0'" in line
            assert "inputs.release_tag == 'v0.15.0'" in line
    for version,revision in [('0.14.0','0cb1358a18fb2b7b3c619fb449ae3f058aea3f8d'),('0.14.1','1ffb86d1bd5e3575a00b245d594b0f705e518689'),('0.14.2','a1a138bb34e7ae6d27cc6193d978cf30cbcfa2eb')]:
        assert revision in workflow
        assert f'/releases/download/v{version}/welt_client-{version}-py3-none-any.whl' in workflow
    assert 'site/v0.15.0/backtesting.html' in workflow
    assert 'site/v0.14.2/downloads/forecasting_covariates.py' in workflow


def test_current_builder_preserves_previous_page_and_download_bytes_and_navigation():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);sentinels={}
        for old in ['0.14.0','0.14.1','0.14.2','0.15.0']:
            for name in ['index.html','forecasting.html','backtesting.html','downloads/forecasting_covariates.py','downloads/forecasting-with-covariates.ipynb']:
                p=root/f'v{old}'/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(f'immutable {old} {name}\n'.encode());sentinels[p]=p.read_bytes()
        with patch.object(sys,'argv',['build_docs','--output',folder]):build_docs.main()
        for p,expected in sentinels.items():assert p.read_bytes()==expected
        manifest=json.loads((root/'v0.16.0/manifest.json').read_text())
        assert {'0.14.0','0.14.1','0.14.2','0.15.0'}<=set(manifest['historical_versions'])
        for old in ['0.14.0','0.14.1','0.14.2','0.15.0']:
            assert f'/v{old}/</loc>' in (root/'sitemap.xml').read_text()
            assert f'value="../v{old}/"' in (root/'v0.16.0/index.html').read_text()
        assert (root/'v0.16.0/backtesting.html').is_file()

def test_benchmark_release_retains_matching_old_source_and_published_wheel():
    workflow=(build_docs.ROOT/'.github/workflows/sdk-docs.yml').read_text()
    assert 'df590cf6bc5fdf38cf4229f5327719784e6233b1' in workflow
    assert '/releases/download/v0.15.0/welt_client-0.15.0-py3-none-any.whl' in workflow
    assert 'site/v0.16.0/backtesting.html' in workflow
    assert 'Backtest.feature_selection' in workflow
