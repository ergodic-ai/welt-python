"""Execute the published forecasting guide, script and notebook against the DEC-085 wire fixture.

API-162. The fixture returns seasonal-naive numbers, so these tests check SDK usage,
request shapes and the tutorial's code paths, never Chronos-2 forecast quality.
"""
import json
from pathlib import Path
import re
import runpy

import httpx
import pytest

from welt import AsyncClient, Client, ForecastResult
from scripts.notebook_fixture import ForecastSyntheticService

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / 'docs/forecasting.md'


@pytest.fixture
def service(monkeypatch):
    fixture = ForecastSyntheticService()
    for kind in (Client, AsyncClient):
        original = kind.__init__

        def initialize(self, *args, _original=original, **kwargs):
            kwargs.update(base_url='https://notebook.invalid', api_key='test-only-key',
                          transport=httpx.MockTransport(fixture))
            _original(self, *args, **kwargs)
        monkeypatch.setattr(kind, '__init__', initialize)
    return fixture


def steps():
    """The guide's step programs: python fences labelled file="forecast_sales.py", in order."""
    return re.findall(r'```python file="forecast_sales.py"\n(.*?)```', GUIDE.read_text(), re.S)


def test_guide_steps_run_in_order_as_one_script(service, capsys):
    programs = steps()
    assert len(programs) >= 10
    namespace = {}
    for program in programs:
        exec(compile(program, 'forecast_sales.py', 'exec'), namespace)
    capsys.readouterr()  # Fixture numbers are not forecast quality.
    posts = service.posts
    # One series, panel, two targets, covariates, quantiles, cross-learning, the job form.
    assert len(posts) == 7
    one, panel, targets, covariates, quantiles, shared, job = posts
    assert 'id_column' not in one and one['targets'] == ['sales']
    assert panel['id_column'] == 'store' and 'future' not in panel
    assert targets['targets'] == ['sales', 'visits']
    assert covariates['past_covariates'] == ['temperature']
    assert covariates['future_covariates'] == ['promo', 'holiday']
    assert covariates['categorical_covariates'] == ['holiday']
    assert covariates['future']['table']['columns'] == ['date', 'store', 'promo', 'holiday']
    assert len(covariates['future']['table']['rows']) == 3 * 14
    assert quantiles['quantile_levels'] == [0.05, 0.1, 0.5, 0.9, 0.95]
    assert shared['cross_learning'] is True and covariates['cross_learning'] is False
    assert job == covariates
    frame = namespace['frame']
    assert list(frame.columns) == ['id', 'target', 'timestamp', 'median', 'q_0.05', 'q_0.1', 'q_0.5',
                                   'q_0.9', 'q_0.95']
    # The pricing example uses the documented formula on the step 4 request.
    history_rows = len(covariates['history']['table']['rows'])
    variates = 2 + 1 + 2
    assert namespace['tokens'] == (history_rows + 3 * 14) * variates == 5685
    assert isinstance(namespace['result'], ForecastResult)
    assert namespace['daily'].frequency_ == {'unit': 'day', 'every': 1}
    assert namespace['job'].client.http.is_closed  # the estimator-owned job client was closed


def test_guide_documents_release_placeholders_and_honest_intervals():
    text = GUIDE.read_text()
    assert 'TBD-API-160' in text and '<!-- release:' in text
    assert 'not calibrated' in text and 'rearranged_steps' in text and 'monotone' in text
    assert '50 credits per\nmillion tokens' in text or '50 credits per million tokens' in text
    assert 'https://docs.ergodic.dev/forecast-with-welt.html' in text
    # Limits are read from metadata in code, never hard-coded in the step programs.
    assert 'client.model("chronos-2")' in text


def test_covariate_script_runs_end_to_end(service, capsys):
    runpy.run_path(str(ROOT / 'examples/forecasting_covariates.py'), run_name='__main__')
    out = capsys.readouterr().out
    assert 'MASE' in out and 'coverage' in out
    (request,) = service.posts
    assert request['future_covariates'] == ['promo', 'holiday'] and request['past_covariates'] == ['temperature']
    assert request['categorical_covariates'] == ['holiday'] and request['horizon'] == 28


def test_forecasting_notebook_cells_run_in_order(service, capsys):
    try:
        import matplotlib
        matplotlib.use('Agg')
        plotting = True
    except ImportError:  # matplotlib is not an SDK dependency; Colab ships it.
        plotting = False
    notebook = json.loads((ROOT / 'examples/notebooks/forecasting-with-covariates.ipynb').read_text())
    cells = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert cells[0] == '!pip install "welt-client[parquet]"'
    assert sum('matplotlib' in cell for cell in cells) == 1
    namespace = {}
    for cell in cells[1:]:
        if 'matplotlib' in cell and not plotting:
            continue
        exec(compile(cell, 'forecasting notebook', 'exec'), namespace)
    capsys.readouterr()
    (request,) = service.posts
    assert request['horizon'] == 28 and len(request['future']['table']['rows']) == 5 * 28
    # The holdout is chronological: every held-out day is after every history day.
    assert namespace['history']['date'].max() < namespace['test']['date'].min()
    assert set(namespace['scores'].index) >= {'seasonal naive', 'Chronos-2'}
