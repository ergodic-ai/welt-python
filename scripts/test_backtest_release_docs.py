"""API-195: retain the released documentation while publishing backtesting."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from scripts import build_docs


def test_release_workflow_builds_old_sources_and_exact_published_wheels():
    workflow=(build_docs.ROOT/'.github/workflows/sdk-docs.yml').read_text()
    assert 'default: v0.16.2' in workflow
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
        for old in ['0.14.0','0.14.1','0.14.2','0.15.0','0.16.0','0.16.1']:
            for name in ['index.html','forecasting.html','backtesting.html','downloads/forecasting_covariates.py','downloads/forecasting-with-covariates.ipynb']:
                p=root/f'v{old}'/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(f'immutable {old} {name}\n'.encode());sentinels[p]=p.read_bytes()
        with patch.object(sys,'argv',['build_docs','--output',folder]):build_docs.main()
        for p,expected in sentinels.items():assert p.read_bytes()==expected
        manifest=json.loads((root/'v0.16.2/manifest.json').read_text())
        assert {'0.14.0','0.14.1','0.14.2','0.15.0','0.16.0','0.16.1'}<=set(manifest['historical_versions'])
        for old in ['0.14.0','0.14.1','0.14.2','0.15.0','0.16.0','0.16.1']:
            assert f'/v{old}/</loc>' in (root/'sitemap.xml').read_text()
            assert f'value="../v{old}/"' in (root/'v0.16.2/index.html').read_text()
        assert (root/'v0.16.2/backtesting.html').is_file()

def test_benchmark_release_retains_matching_old_source_and_published_wheel():
    workflow=(build_docs.ROOT/'.github/workflows/sdk-docs.yml').read_text()
    assert 'df590cf6bc5fdf38cf4229f5327719784e6233b1' in workflow
    assert '/releases/download/v0.15.0/welt_client-0.15.0-py3-none-any.whl' in workflow
    assert 'site/v0.16.0/backtesting.html' in workflow
    assert 'Backtest.feature_selection' in workflow


def test_lookback_patch_retains_immutable_016_and_documents_effective_context():
    workflow=(build_docs.ROOT/'.github/workflows/sdk-docs.yml').read_text()
    assert 'f3a6a3496f8bdda1a38d6cfa23c2c31e041d4e97' in workflow
    assert '/releases/download/v0.16.0/welt_client-0.16.0-py3-none-any.whl' in workflow
    text=(build_docs.ROOT/'docs/backtesting.md').read_text()
    for field in ['single-scale-lookback64-v1','lookback=64','effective_context_length','effective_train_start','effective_train_stop','MASE','shared available fold']:
        assert field in text
    assert 'site/v0.16.1/backtesting.html' in workflow


def test_report_documentation_preserves_preceding_client_and_guide_source():
    workflow = (build_docs.ROOT / '.github/workflows/sdk-docs.yml').read_text()
    assert '9f8ce89a20676be3397a767fca6c616d69f16f91' in workflow
    assert 'releases/download/v0.16.1/welt_client-0.16.1-py3-none-any.whl' in workflow
    guide = (build_docs.ROOT / 'docs/backtesting.md').read_text()
    assert 'declared-seasonality-lookback64-v1' in guide
    assert 'report-links' in guide
    assert 'does not authorize model execution' in guide
