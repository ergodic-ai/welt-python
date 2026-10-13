"""Evaluation submission, polling, typed results and report display; controlled HTTP only."""
import asyncio
from copy import deepcopy
import json
import re
import warnings

import httpx
import numpy as np
import pandas as pd
import pytest

from welt import (AsyncClient, AsyncEvaluation, Classifier, Client, ConflictError, Evaluation, EvaluationWarning,
                  ExecutionError, InvalidInputError, JobCancelledError, JobTimeoutError, Regressor, ReportLink,
                  ResultExpiredError, TransportError, WeltError)
from welt.evaluation import _scoped, notebook_html

CLASSIFICATION = ("roc_auc", "log_loss", "accuracy")
REGRESSION = ("rmse", "mae", "r2")
REPORT = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><title>Churn → label · Welt evaluation</title>'
          '<style>html,body{margin:0;padding:0;background:#fafafa}.welt-report{min-height:100vh}</style></head><body>'
          '<div class="welt-report"><style>.welt-report{--wr-bg:#fafafa;color:var(--wr-fg)}'
          '@media (prefers-color-scheme: dark){.welt-report{--wr-bg:#0f0f11}}'
          '.welt-report,.welt-report *{box-sizing:border-box}.welt-report h1,.welt-report p{margin:0}</style>'
          '<main class="wr-main"><h1>Churn &lt;b&gt;</h1><svg viewBox="0 0 1 1"></svg></main></div></body></html>\n')


def scores(metrics, value):
    return {m: (None if m == "log_loss" and value <= 0.5 else value) for m in metrics}


def entry(model_id, name, value, metrics):
    return dict(model_id=model_id, name=name, mean=scores(metrics, value), sd=scores(metrics, value / 10),
                pooled=scores(metrics, value))


class Service:
    """Evaluation routes generated in memory, solely to verify client behavior."""

    def __init__(self, task="classification", outcome="succeeded", polls=2):
        self.task, self.outcome, self.polls = task, outcome, polls
        self.requests, self.datasets, self.evaluations, self.keys, self.links = [], {}, {}, {}, {}
        self.reads = 0
        self.expired = False
        self.lose_response = False
        self.error = None
        self.report = REPORT

    def resource(self, body, identity):
        metrics = CLASSIFICATION if body["task"] == "classification" else REGRESSION
        models = body["models"] or ["tabicl-v2", "tabdpt-1.3"]
        return dict(
            id=identity, status="queued", stage="queued", operation_id="op_1", dataset_id=body["dataset_id"],
            dataset_name="SDK dataset", task=body["task"], target="label", rows=6, dataset_rows=6,
            feature_columns=["x1", "x2"], folds=body["folds"] or 3, split=body["split"], seed=body["seed"],
            group_column=body["group_column"], fold_column=body["fold_column"], sample_rows=None,
            primary_metric=body["primary_metric"] or metrics[0],
            positive_class="yes" if body["task"] == "classification" else None,
            classes=["no", "yes"] if body["task"] == "classification" else None,
            class_counts=[3, 3] if body["task"] == "classification" else None, fold_sha256="0" * 64,
            protocol_version="evaluation-protocol-v1", metrics_version="metrics-v1", report_version="report-v1",
            model_selection="explicit" if body["models"] else "evaluation-defaults-v1",
            models=[dict(model_id=m, name=m.upper(), model_version=m + "-v1", configuration="default",
                         resolved_configuration={}, job_id=f"job_{i}", status="queued", error=None)
                    for i, m in enumerate(models)],
            skipped=[] if body["models"] else [dict(model_id="kumo-medium", name="Kumo", reason="workload_limit",
                                                    message="Too many rows.")],
            estimated_credits=12, summary=None, report_available=False, predictions_expires_at=None,
            error=None, created_at="2026-10-13T10:00:00+00:00", updated_at="2026-10-13T10:00:00+00:00",
            completed_at=None)

    def finish(self, evaluation):
        metrics = CLASSIFICATION if evaluation["task"] == "classification" else REGRESSION
        evaluation.update(status=self.outcome, stage="complete")
        for i, model in enumerate(evaluation["models"]):
            failed = self.outcome == "failed" or (self.outcome == "partially_completed" and i == 1)
            model["status"] = {"cancelled": "cancelled"}.get(self.outcome, "failed" if failed else "succeeded")
            if failed:
                model["error"] = dict(code="execution_failed", message="Evaluating this model failed.")
        if self.outcome in ("succeeded", "partially_completed"):
            ranked = [m for m in evaluation["models"] if m["status"] == "succeeded"]
            evaluation.update(report_available=True, predictions_expires_at="2026-10-20T10:00:00+00:00", summary=dict(
                primary_metric=evaluation["primary_metric"], metrics=list(metrics),
                directions={m: "lower" if m in ("log_loss", "rmse", "mae") else "higher" for m in metrics},
                leaderboard=[entry(m["model_id"], m["name"], 0.9 - 0.1 * i, metrics) for i, m in enumerate(ranked)],
                baseline=entry("baseline", "Baseline", 0.5, metrics),
                comparison=dict(metric=metrics[0], first=ranked[0]["model_id"], separated=False) if len(ranked) > 1
                else None, warnings=[dict(code="small_dataset", message="Fewer than 100 rows.")]))
        elif self.outcome == "failed":
            evaluation["error"] = dict(code="execution_failed", message="No model finished; see each model's error.")

    def __call__(self, request):
        self.requests.append(request)
        path, method = request.url.path, request.method
        body = json.loads(request.content) if request.content else None
        if self.error and path == "/v1/evaluations" and method == "POST":
            return httpx.Response(self.error[0], json={"error": self.error[1]})
        if path == "/v1/datasets" and method == "POST":
            identity = f"ds_{len(self.datasets)}"
            self.datasets[identity] = body
            return httpx.Response(201, json=dict(id=identity, **body))
        if path == "/v1/evaluations" and method == "POST":
            key = request.headers["Idempotency-Key"]
            if key in self.keys:
                previous, identity = self.keys[key]
                if previous != body:
                    return httpx.Response(409, json={"error": {"code": "idempotency_conflict", "message": "Key used."}})
                return httpx.Response(202, json=self.evaluations[identity])
            identity = f"ev_{len(self.evaluations) + 1}"
            self.evaluations[identity] = self.resource(body, identity)
            self.keys[key] = (body, identity)
            if self.lose_response:
                self.lose_response = False
                raise httpx.ConnectError("response lost", request=request)
            return httpx.Response(202, json=self.evaluations[identity])
        if path == "/v1/evaluations" and method == "GET":
            return httpx.Response(200, json={"items": list(self.evaluations.values())})
        if path == "/v1/fits":
            return httpx.Response(202, json=dict(id="job_fit", operation="fit", status="queued"))
        if path == "/v1/jobs/job_fit":
            return httpx.Response(200, json=dict(id="job_fit", operation="fit", status="succeeded", predictor_id="pred_1"))
        if path == "/v1/predictors/pred_1":
            fit = json.loads([r.content for r in self.requests if r.url.path == "/v1/fits"][-1])
            return httpx.Response(200, json=dict(id="pred_1", model_id=fit["model_id"], model_version="v1",
                configuration=fit["configuration"], seed=fit["seed"], task=fit["task"], columns=["x1", "x2"],
                classes=["no", "yes"] if fit["task"] == "classification" else None))
        parts = path.split("/")
        evaluation = self.evaluations.get(parts[3]) if len(parts) > 3 else None
        if evaluation is None:
            return httpx.Response(404, json={"error": {"code": "not_found", "message": "Resource not found."}})
        tail = "/".join(parts[4:])
        if tail == "" and method == "GET":
            self.reads += 1
            if evaluation["status"] in ("queued", "running"):
                if self.reads >= self.polls:
                    self.finish(evaluation)
                else:
                    evaluation.update(status="running", stage="1 of 2 models complete")
            return httpx.Response(200, json=evaluation)
        if tail == "" and method == "DELETE":
            del self.evaluations[parts[3]]
            return httpx.Response(204)
        if tail == "cancel":
            evaluation.update(status="cancelled", stage="cancelled")
            return httpx.Response(200, json=evaluation)
        if evaluation["status"] not in ("succeeded", "partially_completed"):
            code = "result_not_ready" if evaluation["status"] in ("queued", "running") else "result_unavailable"
            return httpx.Response(409, json={"error": {"code": code, "message": "The evaluation has no results."}})
        ranked = [e["model_id"] for e in evaluation["summary"]["leaderboard"]] + ["baseline"]
        if tail == "folds":
            metrics = evaluation["summary"]["metrics"]
            return httpx.Response(200, json=dict(evaluation_id=evaluation["id"], metrics=metrics, items=[
                dict(model_id=m, fold=f, rows=2, values={k: (None if k == "log_loss" else 0.1 * f) for k in metrics})
                for m in ranked for f in range(evaluation["folds"])]))
        if tail == "predictions":
            if self.expired:
                return httpx.Response(410, json={"error": {"code": "result_expired", "message": "Expired."}})
            classification = evaluation["task"] == "classification"
            actual = ["no", "yes"] * 3 if classification else [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            return httpx.Response(200, json=dict(
                evaluation_id=evaluation["id"], task=evaluation["task"], classes=evaluation["classes"],
                row_index=[0, 1, 2, 3, 4, 5], fold=[0, 1, 2, 0, 1, 2], actual=actual, expires_at="2026-10-20",
                models={m: dict(predictions=actual, probabilities=[[0.8, 0.2] if a == "no" else [0.3, 0.7]
                                                                   for a in actual] if classification else None)
                        for m in ranked}))
        if tail == "report":
            return httpx.Response(200, text=self.report, headers={"content-type": "text/html; charset=utf-8"})
        if tail == "report-links" and method == "POST":
            identity = f"rl_{len(self.links) + 1}"
            self.links[identity] = dict(id=identity, evaluation_id=evaluation["id"], created_at="2026-10-13",
                                        expires_at="2026-10-20", revoked=False, created_by="key_1")
            return httpx.Response(201, json=dict(self.links[identity], url=f"https://welt.test/reports/secret{identity}"))
        if tail == "report-links" and method == "GET":
            return httpx.Response(200, json={"items": list(self.links.values())})
        if tail.startswith("report-links/") and method == "DELETE":
            self.links[parts[-1]]["revoked"] = True
            return httpx.Response(204)
        raise AssertionError("unexpected route " + method + " " + path)

    def routes(self):
        return [(r.method, r.url.path) for r in self.requests]


def table(n=6):
    return pd.DataFrame(dict(x1=np.arange(n, dtype=float), x2=np.linspace(0, 1, n), label=["no", "yes"] * (n // 2)))


@pytest.fixture
def service():
    return Service()


def client_for(service):
    return Client(api_key="fixture-only", transport=httpx.MockTransport(service))


def test_table_upload_keeps_target_and_full_body_with_generated_key(service):
    with client_for(service) as client:
        evaluation = client.submit_evaluation(table(), "label", task="classification", models=["tabicl-v2"],
                                              folds=3, seed=7, positive_class=np.str_("yes"), max_credits=50)
    assert isinstance(evaluation, Evaluation) and evaluation.id == "ev_1" and evaluation.status == "queued"
    upload = service.datasets["ds_0"]
    assert upload["columns"] == ["x1", "x2", "label"] and upload["target"] == "label"
    assert upload["rows"][1] == [1.0, 0.2, "yes"]
    post = service.requests[-1]
    assert json.loads(post.content) == dict(dataset_id="ds_0", task="classification", models=["tabicl-v2"], folds=3,
        split="auto", group_column=None, fold_column=None, seed=7, primary_metric=None, positive_class="yes",
        sample_rows=None, max_credits=50)
    assert re.fullmatch(r"[0-9a-f]{32}", post.headers["Idempotency-Key"])
    assert evaluation.estimated_credits == 12 and evaluation.models[0]["model_id"] == "tabicl-v2"


def test_arrays_files_and_stored_datasets(service, tmp_path):
    path = tmp_path / "train.csv"
    table().to_csv(path, index=False)
    with client_for(service) as client:
        client.submit_evaluation(np.c_[np.arange(6.0), np.ones(6)], np.arange(6.0), task="regression")
        client.submit_evaluation(str(path), "label", task="classification")
        stored = client.submit_evaluation("ds_existing", task="classification", split="group", group_column="g")
    assert service.datasets["ds_0"]["columns"] == ["x0", "x1", "target"]
    assert service.datasets["ds_1"]["columns"] == ["x1", "x2", "label"]
    assert stored.dataset_id == "ds_existing" and len(service.datasets) == 2
    assert stored.skipped == [dict(model_id="kumo-medium", name="Kumo", reason="workload_limit", message="Too many rows.")]


@pytest.mark.parametrize("arguments,message", [
    (dict(task="ranking"), "task"), (dict(task="classification", models="tabicl-v2"), "list"),
    (dict(task="classification", models=["a"] * 2), "unique"), (dict(task="classification", folds=11), "folds"),
    (dict(task="classification", split="group"), "group_column"),
    (dict(task="classification", fold_column="fold"), "custom"),
    (dict(task="regression", split="stratified"), "Stratified"), (dict(task="classification", seed=-1), "seed"),
    (dict(task="classification", sample_rows=5), "sample_rows"),
    (dict(task="regression", positive_class="yes"), "positive_class"),
    (dict(task="classification", idempotency_key="bad key"), "idempotency_key"),
])
def test_options_are_refused_before_any_upload(service, arguments, message):
    with client_for(service) as client, pytest.raises(ValueError, match=message):
        client.submit_evaluation(table(), "label", **arguments)
    assert service.requests == []


def test_targets_and_split_columns_are_checked_locally(service):
    with client_for(service) as client:
        with pytest.raises(ValueError, match="numeric"):
            client.submit_evaluation(table(), "label", task="regression")
        with pytest.raises(ValueError, match="Split columns"):
            client.submit_evaluation(table(), "label", task="classification", split="group", group_column="absent")
        with pytest.raises(ValueError, match="own target"):
            client.submit_evaluation("ds_existing", "label", task="classification")
        with pytest.raises(ValueError, match="target"):
            client.submit_evaluation(table(), task="classification")
    assert service.requests == []


def test_caller_replay_with_the_same_key_reconnects_after_a_lost_response(service):
    service.lose_response = True
    with client_for(service) as client:
        with pytest.raises(TransportError) as lost:
            client.submit_evaluation("ds_existing", task="classification", idempotency_key="eval-run-0001")
        assert "submit_evaluation('ds_existing'" in str(lost.value) and "eval-run-0001" in str(lost.value)
        replayed = client.submit_evaluation("ds_existing", task="classification", idempotency_key="eval-run-0001")
        changed = pytest.raises(ConflictError, client.submit_evaluation, "ds_existing", task="classification",
                                seed=1, idempotency_key="eval-run-0001")
    posts = [r for r in service.requests if r.url.path == "/v1/evaluations"]
    assert [r.headers["Idempotency-Key"] for r in posts] == ["eval-run-0001"] * 3
    assert json.loads(posts[0].content) == json.loads(posts[1].content)
    assert replayed.id == "ev_1" and list(service.evaluations) == ["ev_1"]
    assert changed.value.code == "idempotency_conflict"


def test_wait_polls_with_backoff_until_finished(service, monkeypatch):
    sleeps = []
    monkeypatch.setattr("welt.evaluation.time.sleep", sleeps.append)
    service.polls = 4
    with client_for(service) as client:
        evaluation = client.evaluate(table(), "label", task="classification")
    assert evaluation.status == "succeeded" and evaluation.finished
    assert service.reads == 4 and len(sleeps) == 3 and sleeps[1] > sleeps[0]
    assert evaluation.comparison["separated"] is False and evaluation.warnings[0]["code"] == "small_dataset"


def test_local_timeout_names_the_evaluation_and_never_cancels(service):
    service.polls = 10**6
    with client_for(service) as client:
        evaluation = client.submit_evaluation("ds_existing", task="classification")
        with pytest.raises(JobTimeoutError) as timeout:
            evaluation.wait(timeout=0.3, poll_interval=0.05)
    assert timeout.value.job_id == "ev_1" and "Client().evaluation('ev_1').wait()" in str(timeout.value)
    assert all(not path.endswith("/cancel") for _, path in service.routes())


def test_classification_frames(service):
    service.polls = 1
    with client_for(service) as client:
        evaluation = client.evaluate("ds_existing", task="classification")
        summary, folds, predictions = evaluation.summary(), evaluation.folds(), evaluation.predictions()
    assert list(summary.index) == ["tabicl-v2", "tabdpt-1.3", "baseline"] and summary.index.name == "model_id"
    assert list(summary.columns) == ["name", "rank"] + [f"{m}_{s}" for m in CLASSIFICATION for s in ("mean", "sd")]
    assert summary["rank"].tolist()[:2] == [1, 2] and pd.isna(summary.loc["baseline", "rank"])
    assert summary.loc["tabicl-v2", "roc_auc_mean"] == pytest.approx(0.9)
    assert summary.loc["tabicl-v2", "roc_auc_sd"] == pytest.approx(0.09)
    assert np.isnan(summary.loc["baseline", "log_loss_mean"])
    assert list(folds.columns) == ["model_id", "fold", "rows", "metric", "value"]
    assert len(folds) == 3 * 3 * 3 and folds["fold"].dtype == np.int64 and folds["value"].dtype == np.float64
    assert folds.query("metric == 'log_loss'")["value"].isna().all()
    assert list(predictions.columns[:3]) == ["row_index", "fold", "actual"]
    assert ["tabicl-v2", "tabicl-v2:p(no)", "tabicl-v2:p(yes)"] == list(predictions.columns[3:6])
    assert "baseline:p(yes)" in predictions and predictions["tabicl-v2:p(yes)"].tolist()[:2] == [0.2, 0.7]
    assert predictions["actual"].tolist() == ["no", "yes"] * 3


def test_regression_frames(service):
    service.polls = 1
    with client_for(service) as client:
        evaluation = client.evaluate("ds_existing", task="regression", primary_metric="mae")
        summary, predictions = evaluation.summary(), evaluation.predictions()
    assert evaluation.primary_metric == "mae" and "rmse_mean" in summary and "r2_sd" in summary
    assert list(predictions.columns) == ["row_index", "fold", "actual", "tabicl-v2", "tabdpt-1.3", "baseline"]
    assert predictions["tabicl-v2"].dtype == np.float64


def test_results_before_finishing_expiry_and_refusals(service):
    with client_for(service) as client:
        evaluation = client.submit_evaluation("ds_existing", task="classification")
        with pytest.raises(ConflictError) as early:
            evaluation.summary()
        assert early.value.code == "result_not_ready" and early.value.retryable
        with pytest.raises(ConflictError) as remote:
            evaluation.folds()
        assert remote.value.code == "result_not_ready"
        evaluation.refresh().refresh()
        service.expired = True
        with pytest.raises(ResultExpiredError) as expired:
            evaluation.predictions()
        assert expired.value.code == "result_expired" and expired.value.status_code == 410
        assert evaluation.summary().shape[0] == 3 and len(evaluation.folds())  # Summary and folds remain.
        service.error = (422, dict(code="budget_exceeded", message="Estimated at 12 credits, above 5.",
                                   details=dict(estimated_credits=12, max_credits=5)))
        with pytest.raises(InvalidInputError) as budget:
            client.submit_evaluation("ds_existing", task="classification", max_credits=5)
        assert budget.value.code == "budget_exceeded" and budget.value.details == dict(estimated_credits=12, max_credits=5)
        service.error = (422, dict(code="no_eligible_models", message="No available model can evaluate this data.",
                                   details=dict(skipped=[dict(model_id="tabicl-v2", reason="workload_limit")])))
        with pytest.raises(InvalidInputError) as none:
            client.submit_evaluation("ds_existing", task="classification")
        assert none.value.details["skipped"][0]["reason"] == "workload_limit"


def test_error_details_redact_the_configured_key():
    def service(request):
        return httpx.Response(422, json={"error": {"code": "no_eligible_models", "message": "None.",
                                                   "details": {"skipped": [{"message": "key fixture-only"}]}}})
    with Client(api_key="fixture-only", transport=httpx.MockTransport(service)) as client:
        with pytest.raises(InvalidInputError) as error:
            client.submit_evaluation("ds_existing", task="classification")
    assert error.value.details == {"skipped": [{"message": None}]}


@pytest.mark.parametrize("outcome,error,code", [("failed", ExecutionError, "execution_failed"),
                                                 ("cancelled", JobCancelledError, "job_cancelled")])
def test_failed_and_cancelled_evaluations_raise_typed_errors(outcome, error, code):
    service = Service(outcome=outcome, polls=1)
    with client_for(service) as client, pytest.raises(error) as raised:
        client.evaluate("ds_existing", task="classification")
    assert raised.value.code == code and raised.value.job_id == "ev_1"


def test_partial_evaluation_returns_with_a_warning():
    service = Service(outcome="partially_completed", polls=1)
    with client_for(service) as client, pytest.warns(EvaluationWarning, match="TABDPT-1.3 did not finish"):
        evaluation = client.evaluate("ds_existing", task="classification")
    assert evaluation.status == "partially_completed" and list(evaluation.summary().index) == ["tabicl-v2", "baseline"]
    assert evaluation.models[1]["error"]["code"] == "execution_failed"


def test_cancel_delete_list_and_reopen(service):
    with client_for(service) as client:
        evaluation = client.submit_evaluation("ds_existing", task="classification")
        assert evaluation.cancel() is evaluation and evaluation.status == "cancelled"
        reopened = client.evaluation("ev_1")
        assert reopened.to_dict() == evaluation.to_dict() and [e.id for e in client.evaluations()] == ["ev_1"]
        assert evaluation.delete() is None and service.routes()[-1] == ("DELETE", "/v1/evaluations/ev_1")
        with pytest.raises(WeltError) as missing:
            client.evaluation("ev_1")
    assert missing.value.status_code == 404


def test_hostile_identities_are_refused():
    payload = Service().resource(dict(dataset_id="ds", task="classification", models=[], folds=None, split="auto",
                                      seed=0, group_column=None, fold_column=None, primary_metric=None), "ev_1")
    def service(request):
        return httpx.Response(200, json=dict(payload, id="ev_other") if "ev_1" in request.url.path
                              else dict(payload, id="ev_fixture-only"))
    with Client(api_key="fixture-only", transport=httpx.MockTransport(service)) as client:
        with pytest.raises(WeltError, match="different evaluation"):
            client.evaluation("ev_1")
        with pytest.raises(WeltError, match="Invalid evaluation"):
            client.evaluation("ev_2")
    with Client(api_key="k") as client, pytest.raises(WeltError, match="Invalid evaluation"):
        Evaluation(client, dict(payload, status="succeeded"))  # Finished without a summary.


def test_report_html_and_links(service):
    service.polls = 1
    with client_for(service) as client:
        evaluation = client.evaluate("ds_existing", task="classification")
        assert evaluation.report_html() == REPORT and evaluation.report_html() == REPORT
        link = evaluation.report_link(days=3)
        assert isinstance(link, ReportLink) and link.url == "https://welt.test/reports/secretrl_1"
        assert link.expires_at == "2026-10-20" and link.revoked is False and "secret" not in repr(link)
        listed = evaluation.report_links()
        evaluation.revoke_report_link(link)
        evaluation.revoke_report_link("rl_1")
        for days in (0, 8, 2.0, True):
            with pytest.raises(ValueError, match="days"):
                evaluation.report_link(days=days)
    assert [r.url.path for r in service.requests].count("/v1/evaluations/ev_1/report") == 1
    post = next(r for r in service.requests if r.method == "POST" and r.url.path.endswith("/report-links"))
    assert json.loads(post.content) == {"expires_in_days": 3}
    assert listed == [ReportLink(id="rl_1", evaluation_id="ev_1", url=None, created_at="2026-10-13",
                                 expires_at="2026-10-20", revoked=False, created_by="key_1")]
    assert service.routes()[-1] == ("DELETE", "/v1/evaluations/ev_1/report-links/rl_1")


def test_report_route_must_return_html():
    def service(request):
        return httpx.Response(200, json={"not": "html"})
    with Client(api_key="fixture-only", transport=httpx.MockTransport(service)) as client:
        with pytest.raises(WeltError) as invalid:
            client.request("GET", "/v1/evaluations/ev_1/report", _text=True)
    assert invalid.value.code == "invalid_response"


def stylesheets(markup):
    return re.findall(r"<style[^>]*>(.*?)</style>", markup, re.S)


def test_notebook_html_uses_only_the_scoped_report_body(service):
    service.polls = 1
    with client_for(service) as client:
        evaluation = client.evaluate("ds_existing", task="classification")
        markup = evaluation._repr_html_()
    assert markup.startswith('<div class="welt-report">') and markup.endswith("</div>")
    assert "<html" not in markup and "<head" not in markup and "html,body" not in markup
    assert "min-height:100vh" not in markup and "Churn &lt;b&gt;" in markup
    assert stylesheets(markup) and all(_scoped(css) for css in stylesheets(markup))


@pytest.mark.parametrize("css", ["body{margin:0}", ".welt-report{color:red} h1{margin:0}",
                                 ".welt-report a,p{color:red}", "@font-face{font-family:x}",
                                 "@import url(x.css);", ".welt-reporting{color:red}",
                                 ".welt-report{color:red", ".welt-report{a{color:red}}"])
def test_unscoped_report_styles_are_isolated_in_a_sandboxed_iframe(css):
    document = REPORT.replace(".welt-report{--wr-bg:#fafafa;", css + ".welt-report{--wr-bg:#fafafa;")
    markup = notebook_html(document)
    assert markup.startswith('<iframe title="Welt evaluation report" sandbox=""')
    assert "<style" not in markup and "&lt;style&gt;" in markup and 'srcdoc="&lt;!doctype html&gt;' in markup


def test_scope_check_accepts_the_report_patterns():
    assert _scoped(".welt-report{x:1}@media (max-width: 700px){.welt-report .wr-grid{display:block}}"
                   ".welt-report table td:nth-child(2),.welt-report>p{x:2}/* comment */")
    assert notebook_html("<html><body><script>alert(1)</script></body></html>").startswith("<iframe")


def test_unfinished_and_unreachable_reports_show_a_status_table(service):
    with client_for(service) as client:
        evaluation = client.submit_evaluation("ds_existing", task="classification")
        status = evaluation._repr_html_()
        service.polls = 1
        evaluation.wait()
    assert "<style" not in status and "Welt evaluation ev_1" in status and "queued" in status
    finished = evaluation._repr_html_()  # The client is closed: the repr degrades, never raises.
    assert finished.startswith('<div class="welt-evaluation">') and "succeeded" in finished


def test_show_displays_the_report_in_jupyter(service, monkeypatch):
    import IPython.display
    shown = []
    monkeypatch.setattr(IPython.display, "display", lambda value: shown.append(value.data))
    service.polls = 1
    with client_for(service) as client:
        evaluation = client.submit_evaluation("ds_existing", task="classification")
        evaluation.show()
        evaluation.wait().show()
    assert shown[0].startswith('<div class="welt-evaluation">') and shown[1].startswith('<div class="welt-report">')


def test_fit_runs_a_new_full_data_fit_of_the_chosen_model(service):
    service.polls = 1
    with client_for(service) as client:
        evaluation = client.evaluate("ds_existing", task="classification", seed=11)
        model = evaluation.fit()
        with pytest.raises(ValueError, match="ranked models"):
            evaluation.fit("kumo-medium")
        regression = client.evaluate("ds_existing", task="regression", idempotency_key="regression-0001")
        regressor = regression.fit("tabdpt-1.3")
    fits = [json.loads(r.content) for r in service.requests if r.url.path == "/v1/fits"]
    assert fits[0] == dict(dataset_id="ds_existing", model_id="tabicl-v2", task="classification",
                           configuration="default", seed=11)
    assert isinstance(model, Classifier) and model.predictor_id_ == "pred_1" and model.model == "tabicl-v2"
    assert list(model.classes_) == ["no", "yes"] and list(model.feature_names_in_) == ["x1", "x2"]
    assert model.random_state == 11 and "fixture-only" not in repr(model)
    assert isinstance(regressor, Regressor) and fits[1]["model_id"] == "tabdpt-1.3"


def test_fit_refuses_when_the_stored_dataset_has_split_columns(service):
    service.polls = 1
    with client_for(service) as client:
        evaluation = client.evaluate("ds_existing", task="classification", split="group", group_column="customer")
        with pytest.raises(ValueError, match="split column"):
            evaluation.fit()
    assert all(path != "/v1/fits" for _, path in service.routes())


def test_async_parity_and_requests():
    def run_sync():
        service = Service(polls=2)
        with client_for(service) as client:
            evaluation = client.evaluate(table(), "label", task="classification", idempotency_key="parity-0001",
                                         timeout=60)
            output = [evaluation.to_dict(), evaluation.summary(), evaluation.folds(), evaluation.predictions(),
                      evaluation.report_html(), evaluation.report_link(), evaluation.report_links(),
                      evaluation.fit().predictor_id_, [e.id for e in client.evaluations()]]
            assert client.evaluation(evaluation.id).id == evaluation.id
            evaluation.revoke_report_link("rl_1")
            evaluation.delete()
        return output, service.routes()

    async def run_async():
        service = Service(polls=2)
        async with AsyncClient(api_key="fixture-only", transport=httpx.MockTransport(service)) as client:
            evaluation = await client.evaluate(table(), "label", task="classification",
                                               idempotency_key="parity-0001", timeout=60)
            assert isinstance(evaluation, AsyncEvaluation)
            assert evaluation._repr_html_().startswith('<div class="welt-evaluation">')
            output = [evaluation.to_dict(), evaluation.summary(), await evaluation.folds(),
                      await evaluation.predictions(), await evaluation.report_html(), await evaluation.report_link(),
                      await evaluation.report_links(), (await evaluation.fit()).predictor_id_,
                      [e.id for e in await client.evaluations()]]
            assert evaluation._repr_html_().startswith('<div class="welt-report">')
            assert (await client.evaluation(evaluation.id)).id == evaluation.id
            await evaluation.revoke_report_link("rl_1")
            await evaluation.delete()
        return output, service.routes()

    (expected, routes), (actual, async_routes) = run_sync(), asyncio.run(run_async())
    for left, right in zip(expected, actual):
        if isinstance(left, pd.DataFrame):
            pd.testing.assert_frame_equal(left, right)
        else:
            assert left == right
    assert async_routes == routes


def test_async_wait_failure_and_timeout():
    async def run():
        service = Service(outcome="failed", polls=1)
        async with AsyncClient(api_key="fixture-only", transport=httpx.MockTransport(service)) as client:
            with pytest.raises(ExecutionError):
                await client.evaluate("ds_existing", task="classification")
            slow = Service(polls=10**6)
        async with AsyncClient(api_key="fixture-only", transport=httpx.MockTransport(slow)) as client:
            evaluation = await client.submit_evaluation("ds_existing", task="classification")
            with pytest.raises(JobTimeoutError, match="evaluation"):
                await evaluation.wait(timeout=0.2, poll_interval=0.05)
        assert all(not path.endswith("/cancel") for _, path in slow.routes())
    asyncio.run(run())


def test_evaluation_state_is_an_owned_copy(service):
    with client_for(service) as client:
        evaluation = client.submit_evaluation("ds_existing", task="classification")
    evaluation.models[0]["status"] = "tampered"
    evaluation.to_dict()["models"][0]["status"] = "tampered"
    assert evaluation.models[0]["status"] == "queued"
    assert repr(evaluation) == "Evaluation(id='ev_1', status='queued', task='classification')"
