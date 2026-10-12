"""Execute published audit fixes without accounts, hosted data or model jobs.

Synthetic tables/outputs verify SDK contracts only, never prediction quality.
"""
import asyncio
from hashlib import sha256
import inspect
import json
from pathlib import Path
import re
import shlex
import tempfile

import httpx
import numpy as np
import pandas as pd
import pytest

from welt import AsyncClient, Client, WeltError, cli
from scripts.notebook_fixture import CausalSyntheticService, SyntheticService

ROOT = Path(__file__).resolve().parents[1]


def blocks(path, language="python"):
    return re.findall(r"```" + language + r"\n(.*?)```", path.read_text(), re.S)


def iris_shape():
    frame = pd.DataFrame(np.arange(600, dtype=float).reshape(150, 4), columns=[
        "sepal-length", "sepal-width", "petal-length", "petal-width"])
    frame["target"] = np.repeat([0, 1, 2], 50)
    return frame


def test_readme_install_and_all_python_examples_use_current_sdk(monkeypatch, tmp_path, capsys):
    command = shlex.split(blocks(ROOT / "README.md", "sh")[0])
    assert command == ["python", "-m", "pip", "install", "welt-client[parquet]"]
    readme = (ROOT / "README.md").read_text()
    assert "__target__#0" not in readme and "source-candidate" not in readme
    programs = blocks(ROOT / "README.md")
    start_program = blocks(ROOT / "docs/start.md")[1]
    assert programs[1] == start_program
    body = iris_shape().to_parquet(index=False)
    digest = sha256(body).hexdigest()
    detail = dict(id="asset-d839644b6de36fc5cafdda18c5daeca3", name="Test table",
                  content=dict(version=digest, sha256=digest, size_bytes=len(body),
                               media_type="application/vnd.apache.parquet", filename="iris.parquet"),
                  targets=[], splits=[])
    service = SyntheticService()
    requests = []

    def handle(request):
        requests.append(request)
        assert request.url.host == "notebook.invalid"
        if request.url.path == "/v1/workspace":
            return httpx.Response(200, json={"id": "workspace-test"})
        if request.url.path.startswith("/v1/research-datasets/"):
            if request.url.path.endswith("/content"):
                assert request.url.params["version"] == digest
                return httpx.Response(200, stream=httpx.ByteStream(body))
            return httpx.Response(200, json=detail)
        if request.url.path == "/v1/model-runs":
            service.job_records["forecast-test"] = dict(id="forecast-test", status="succeeded", operation="model_run")
            return httpx.Response(202, json={"id": "forecast-test"})
        if request.url.path == "/v1/jobs/forecast-test/result":
            return httpx.Response(200, json={"predictions": [0.0] * 12})
        return service(request)

    initialize = Client.__init__

    def isolated(self, *args, **kwargs):
        kwargs.update(base_url="https://notebook.invalid", api_key="test-only", transport=httpx.MockTransport(handle))
        initialize(self, *args, **kwargs)

    monkeypatch.setattr(Client, "__init__", isolated)
    original_mkdtemp = tempfile.mkdtemp

    def temporary_here(suffix=None, prefix=None, dir=None):
        return original_mkdtemp(suffix=suffix, prefix=prefix, dir=dir or tmp_path)

    monkeypatch.setattr(tempfile, "mkdtemp", temporary_here)
    namespace = {}
    for program in programs:
        exec(compile(program, "README code", "exec"), namespace)
    assert namespace["train"].shape == (120, 5)
    assert namespace["query"].shape == (30, 4)
    assert set(namespace["train"].index).isdisjoint(namespace["test"].index)
    assert len(service.fitted) == 1
    assert namespace["saved"].metadata_path.exists()
    assert namespace["meta"].name == "Test table"
    assert any(request.url.path == "/v1/model-runs" for request in requests)
    capsys.readouterr()  # Fixture scores are not model-quality evidence.


def test_cli_recipe_prepares_target_free_numeric_observations_then_submits(monkeypatch, tmp_path, capsys):
    page = ROOT / "docs/cli.md"
    monkeypatch.chdir(tmp_path)
    data = iris_shape()
    namespace = {"data": data, "target": "target"}
    for program in blocks(page):
        exec(compile(program, "CLI observations preparation", "exec"), namespace)
    payload = json.loads((tmp_path / "observations.json").read_text())
    assert payload["target"] is None
    assert "target" not in payload["columns"]
    assert np.asarray(payload["rows"]).shape == (150, 4)
    assert np.isfinite(np.asarray(payload["rows"])).all()
    assert data.columns[-1] == "target"  # Preparation never mutates the original table.
    service = CausalSyntheticService()
    original = Client
    monkeypatch.setattr(cli, "Client", lambda **kwargs: original(
        base_url="https://notebook.invalid", api_key="test-only", transport=httpx.MockTransport(service)))
    shell = next(program for program in blocks(page, "sh") if "ds_observations" in program)
    lines = [line for line in shell.splitlines() if line.startswith("welt")]
    assert cli.main(shlex.split(lines[0])[1:]) == 0
    uploaded = json.loads(capsys.readouterr().out)
    command = shlex.split(lines[1])[1:]
    assert "--constraints" not in command
    command[command.index("ds_observations")] = uploaded["id"]
    assert cli.main(command) == 0
    output = json.loads(capsys.readouterr().out)
    assert service.data[uploaded["id"]]["target"] is None
    result = service.causal_results[output["id"]]
    assert not any(result["constraints_applied"].values())


@pytest.mark.parametrize("code", ["insufficient_credits", "spend_cap_reached", "plan_past_due"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_documented_billing_errors_are_base_welt_error_without_retry(code, asynchronous):
    page = (ROOT / "docs/async-and-errors.md").read_text()
    assert f'code="{code}"' in page and "Workspace → Billing" in page
    calls = []

    def refuse(request):
        calls.append(request)
        return httpx.Response(402, json={"error": dict(code=code, message="Billing action required", retryable=False)})

    async def run():
        async with AsyncClient(base_url="https://docs.invalid", api_key="test-only", transport=httpx.MockTransport(refuse)) as client:
            with pytest.raises(WeltError) as error:
                await client.submit_fit("ds-test", model="tabicl-v2", task="classification")
            return error.value

    if asynchronous:
        error = asyncio.run(run())
    else:
        with Client(base_url="https://docs.invalid", api_key="test-only", transport=httpx.MockTransport(refuse)) as client:
            with pytest.raises(WeltError) as caught:
                client.submit_fit("ds-test", model="tabicl-v2", task="classification")
            error = caught.value
    assert type(error) is WeltError
    assert (error.status_code, error.code, error.retryable) == (402, code, False)
    assert len(calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_usage_keeps_nullable_and_repeated_accounting_fields(asynchronous):
    events = [dict(operation_id="op-test", status=status, credits=5, rate_card_version="card-test")
              for status in ("queued", "succeeded")]
    events.append(dict(operation_id="op-unpriced", status="succeeded", credits=None, rate_card_version=None))
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"items": events}))

    async def run():
        async with AsyncClient(base_url="https://docs.invalid", api_key="test-only", transport=transport) as client:
            return await client.usage()

    if asynchronous:
        actual = asyncio.run(run())
    else:
        with Client(base_url="https://docs.invalid", api_key="test-only", transport=transport) as client:
            actual = client.usage()
    assert actual == events
    page = (ROOT / "docs/usage.md").read_text()
    assert "do not sum credits across every event" in page
    assert "five-credit minimum" in page and "25-credit research download" in page
    notebook = json.loads((ROOT / "examples/notebooks/artifacts-and-usage.ipynb").read_text())
    source = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "rate_card_version" in source and "credits:null means unpriced" in source


def test_research_default_size_limit_matches_current_download_and_loader():
    for kind in (Client, AsyncClient):
        for method in ("download_research_dataset", "load_research_dataset"):
            assert inspect.signature(getattr(kind, method)).parameters["max_bytes"].default == 128 * 1024 * 1024
    assert "default download size limit is 128 MiB" in (ROOT / "docs/research-datasets.md").read_text()


def test_renamed_guide_sections_keep_existing_bookmark_anchors():
    from scripts.build_docs import render_markdown
    expected = {
        "connect": ["multiple-workspaces-source-candidate"],
        "capabilities": ["resource-completion-source-candidate"],
        "cli": ["file-uploads-and-causal-discovery"],
        "migration": ["select-the-endpoint-explicitly-when-local", "use-a-dataframe-target",
                      "reconnect-after-waiting", "keep-the-release-boundary", "move-from-07-to-08"],
    }
    for name, anchors in expected.items():
        body, _ = render_markdown((ROOT / "docs" / (name + ".md")).read_text())
        actual = re.findall(r'id="([^"]+)"', body)
        for anchor in anchors:
            assert actual.count(anchor) == 1, (name, anchor)
