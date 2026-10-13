"""Cross-validated model comparison (DEC-076): submission, typed results and the report.

An evaluation belongs to the server; these handles borrow the client that created or
reopened them. DataFrames are owned copies built from the server's results.
"""
import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
import html
import math
from pathlib import Path
import re
import time
from uuid import uuid4
from warnings import warn

import numpy as np
import pandas as pd

from .client import _deadline, _failed, _id, _wait
from .credentials import secret
from .errors import (ConflictError, JobTimeoutError, NotFoundError, OptionalDependencyError, TransportError,
                     WeltError, safe_id)

TASKS = ("classification", "regression")
SPLITS = ("auto", "kfold", "stratified", "group", "custom")
STATUSES = ("queued", "running", "succeeded", "partially_completed", "failed", "cancelled")
FINISHED = ("succeeded", "partially_completed")
FILES = (".csv", ".parquet", ".pq")


class EvaluationWarning(UserWarning):
    """An evaluation finished with some models missing from its leaderboard."""


# --- Submission ----------------------------------------------------------------------

def _integer(value, low, high, label, *, optional=True):
    if value is None and optional:
        return None
    if isinstance(value, np.integer):
        value = value.item()
    if type(value) is not int or value < low or (high is not None and value > high):
        bound = f"from {low} to {high}" if high is not None else f"of at least {low}"
        raise ValueError(f"{label} must be an integer {bound}.")
    return value


def _options(task, models, folds, split, group_column, fold_column, seed, primary_metric, positive_class,
             sample_rows, max_credits):
    if task not in TASKS:
        raise ValueError("task must be 'classification' or 'regression'.")
    if isinstance(models, str):
        raise ValueError("models must be a list of model IDs, for example models=['tabicl-v2'].")
    models = [] if models is None else list(models)
    if (len(models) > 8 or len(set(models)) != len(models)
            or any(not isinstance(m, str) or safe_id(m) is None for m in models)):
        raise ValueError("models must be up to 8 unique model IDs; leave it empty to let Welt choose.")
    if split not in SPLITS:
        raise ValueError("split must be one of " + ", ".join(SPLITS) + ".")
    for column, needed in ((group_column, "group"), (fold_column, "custom")):
        if (split == needed) != (column is not None):
            raise ValueError("split='group' needs group_column and split='custom' needs fold_column; "
                             "neither column is used by another split.")
        if column is not None and (not isinstance(column, str) or not column.strip()):
            raise ValueError("Split columns must be nonempty column names.")
    if split == "stratified" and task != "classification":
        raise ValueError("Stratified splits need a classification task.")
    if primary_metric is not None and not isinstance(primary_metric, str):
        raise ValueError("primary_metric must be a metric name such as 'roc_auc' or 'rmse'.")
    if isinstance(positive_class, np.generic):
        positive_class = positive_class.item()
    if positive_class is not None and (task != "classification"
                                       or not isinstance(positive_class, (str, int, float, bool))):
        raise ValueError("positive_class must be one class label of a binary classification target.")
    return dict(task=task, models=models, folds=_integer(folds, 2, 10, "folds"), split=split,
                group_column=group_column, fold_column=fold_column,
                seed=_integer(seed, 0, 2**32 - 1, "seed", optional=False), primary_metric=primary_metric,
                positive_class=positive_class, sample_rows=_integer(sample_rows, 20, 100_000, "sample_rows"),
                max_credits=_integer(max_credits, 1, None, "max_credits"))


def submission(data, target, **options):
    """Validate locally before any request: (dataset_id or None, upload arguments or None, body)."""
    body = _options(**options)
    if isinstance(data, str) and not data.lower().endswith(FILES):
        if target is not None:
            raise ValueError("A stored dataset evaluates its own target column; omit target.")
        if safe_id(data) is None:
            raise ValueError("Pass a dataset ID, a CSV/Parquet path, a DataFrame or an array.")
        return data, None, body
    from .estimators import _training_table, check_targets, read_table
    if isinstance(data, (str, Path)):
        data = read_table(data)
    if isinstance(target, str):
        columns, rows, _, targets = _training_table(data, None, target)
        name = target
    elif target is None:
        raise ValueError("Name the target column with target='column' or pass the target values.")
    else:
        columns, rows, _, targets = _training_table(data, target, None)
        name = "target"
        while name in columns:
            name += "_"
    check_targets(body["task"], targets)
    if any(c is not None and c not in columns for c in (body["group_column"], body["fold_column"])):
        raise ValueError("Split columns must be named feature columns of the table, not the target.")
    values = [v.item() if isinstance(v, np.generic) else v for v in targets]
    return None, dict(columns=columns + [name], rows=[r + [v] for r, v in zip(rows, values)], target=name), body


def key(idempotency_key):
    if idempotency_key is not None and (not isinstance(idempotency_key, str) or safe_id(idempotency_key) is None):
        raise ValueError("idempotency_key must be a bounded opaque identifier.")
    return idempotency_key or uuid4().hex


def unavailable(error, dataset_id=None):
    """A 404 or 405 without Welt's error envelope means the server has no evaluation routes (an
    older server answers the POST with 405 from its console mount)."""
    if isinstance(error, WeltError) and error.status_code in (404, 405) and error.code == "unknown":
        uploaded = f" The table was uploaded as dataset {dataset_id}." if safe_id(dataset_id) else ""
        return NotFoundError("This Welt server does not offer evaluations (no /v1/evaluations route). Use a server "
                             "release with evaluations, or upgrade the deployment." + uploaded,
                             code="evaluations_unavailable", status_code=404)
    return error


def replay_error(dataset_id, idempotency_key):
    """The evaluation may have been accepted; replaying the same key and body reconnects to it."""
    return TransportError(
        "Could not reach Welt API while submitting the evaluation; it may have been accepted. Replay with "
        f"submit_evaluation('{dataset_id}', ...same options..., idempotency_key='{idempotency_key}') "
        "to reconnect instead of starting another.", code="transport_error", retryable=True)


# --- Responses -----------------------------------------------------------------------

def _invalid(what="evaluation"):
    return WeltError(f"Invalid {what} response.", code="invalid_response")


def _validated(payload, expected, api_key):
    try:
        identity = payload["id"]
        if (safe_id(identity) is None or (api_key and api_key in identity)
                or (expected is not None and identity != expected)):
            raise ValueError()
        if (payload["status"] not in STATUSES or payload["task"] not in TASKS
                or not isinstance(payload["models"], list)):
            raise ValueError()
        if payload["status"] in FINISHED and not isinstance(payload.get("summary"), dict):
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise _invalid() from None
    return deepcopy(payload)


def _number(value):
    return math.nan if value is None else float(value)


def _results(payload):
    if payload["status"] not in FINISHED:
        code = "result_not_ready" if payload["status"] in ("queued", "running") else "result_unavailable"
        raise ConflictError("The evaluation has no results" + (" yet; call wait()." if code == "result_not_ready"
                            else "."), code=code, job_id=payload["id"], retryable=code == "result_not_ready")
    return payload["summary"]


def summary_frame(payload):
    summary = _results(payload)
    try:
        metrics = list(summary["metrics"])
        def row(entry, rank):
            values = dict(model_id=entry["model_id"], name=entry["name"], rank=rank)
            for metric in metrics:
                values[f"{metric}_mean"] = _number(entry["mean"].get(metric))
                values[f"{metric}_sd"] = _number(entry["sd"].get(metric))
            return values
        rows = [row(entry, rank) for rank, entry in enumerate(summary["leaderboard"], 1)]
        rows.append(row(summary["baseline"], None))
        frame = pd.DataFrame(rows).set_index("model_id")
    except (KeyError, TypeError, ValueError, AttributeError):
        raise _invalid("evaluation summary") from None
    frame["rank"] = frame["rank"].astype("Int64")
    return frame


def folds_frame(data, evaluation_id):
    try:
        if data["evaluation_id"] != evaluation_id:
            raise ValueError()
        metrics = list(data["metrics"])
        rows = [(item["model_id"], int(item["fold"]), int(item["rows"]), metric, _number(item["values"].get(metric)))
                for item in data["items"] for metric in metrics]
    except (KeyError, TypeError, ValueError, AttributeError):
        raise _invalid("evaluation folds") from None
    frame = pd.DataFrame(rows, columns=["model_id", "fold", "rows", "metric", "value"])
    return frame.astype({"fold": "int64", "rows": "int64", "value": "float64"})


def _labels(classes):
    """Printed class labels for column names; repr() when labels of different types print alike."""
    names = [str(c) for c in classes]
    if len(set(names)) != len(names):
        names = [repr(c) for c in classes]
    if len(set(names)) != len(names):
        raise ValueError()
    return names


def predictions_frame(data, evaluation_id):
    try:
        if data["evaluation_id"] != evaluation_id:
            raise ValueError()
        n = len(data["row_index"])
        frame = dict(row_index=list(data["row_index"]), fold=list(data["fold"]), actual=list(data["actual"]))
        classes = data.get("classes")
        labels = _labels(classes) if data["task"] == "classification" and classes is not None else None
        if len(frame["fold"]) != n or len(frame["actual"]) != n:
            raise ValueError()
        for model_id, output in data["models"].items():
            columns = [model_id]
            values = list(output["predictions"])
            probabilities = output.get("probabilities")
            if len(values) != n:
                raise ValueError()
            if data["task"] == "classification" and probabilities is not None:
                matrix = np.asarray(probabilities, dtype=float).reshape(n, len(classes))
                columns += [f"{model_id}:p({c})" for c in labels]
            if any(c in frame for c in columns):
                raise ValueError()
            frame[model_id] = values
            for j, column in enumerate(columns[1:]):
                frame[column] = matrix[:, j]
    except (KeyError, TypeError, ValueError, AttributeError):
        raise _invalid("evaluation predictions") from None
    return pd.DataFrame(frame)


@dataclass(frozen=True)
class ReportLink:
    """A signed report URL. Anyone holding ``url`` can read the report until it expires or is revoked."""
    id: str
    evaluation_id: str
    url: str | None = field(repr=False)
    created_at: str
    expires_at: str
    revoked: bool
    created_by: str | None = None


def report_link(payload, evaluation_id):
    try:
        if safe_id(payload["id"]) is None or payload["evaluation_id"] != evaluation_id:
            raise ValueError()
        url = payload.get("url")
        if url is not None and not (isinstance(url, str) and url.startswith(("https://", "http://"))):
            raise ValueError()
        return ReportLink(id=payload["id"], evaluation_id=evaluation_id, url=url, created_at=payload["created_at"],
                          expires_at=payload["expires_at"], revoked=payload["revoked"] is True,
                          created_by=payload.get("created_by"))
    except (KeyError, TypeError, ValueError):
        raise _invalid("report link") from None


def _days(days):
    return _integer(days, 1, 7, "days", optional=False)


def _link_id(link):
    value = link.id if isinstance(link, ReportLink) else link
    if not isinstance(value, str) or safe_id(value) is None:
        raise ValueError("Pass a report link ID or ReportLink.")
    return value


# --- Notebook display ----------------------------------------------------------------

# The server's report policy, applied again inside the frame: nothing loads or runs.
_POLICY = ('<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
           'style-src \'unsafe-inline\'; img-src data:">')
_DOCTYPE = re.compile(r"\s*<!doctype[^>]*>", re.I)


def notebook_html(document):
    """The report isolated in a sandboxed iframe for notebook output.

    ``sandbox=""`` (no ``allow-scripts``) stops scripts, forms and navigation of the
    notebook, the frame's own policy blocks every network load, and the report's styles
    cannot reach the notebook page. Nothing from the document is inlined.
    """
    doctype = _DOCTYPE.match(document)
    cut = doctype.end() if doctype else 0
    framed = document[:cut] + _POLICY + document[cut:]
    return ('<div class="welt-evaluation-report"><iframe title="Welt evaluation report" sandbox="" '
            'referrerpolicy="no-referrer" style="width:100%;height:900px;border:0" '
            f'srcdoc="{html.escape(framed, quote=True)}"></iframe></div>')


def status_html(payload, hint=None):
    """A small table for an unfinished evaluation; no stylesheet."""
    e = lambda value: html.escape(str(value))
    rows = [("Status", payload["status"]), ("Stage", payload.get("stage", "")), ("Task", payload["task"]),
            ("Target", payload.get("target", "")), ("Rows", payload.get("rows", "")),
            ("Folds", payload.get("folds", ""))]
    rows += [(m.get("name") or m.get("model_id"), m.get("status")) for m in payload["models"]]
    if payload.get("estimated_credits") is not None:
        rows.append(("Estimated credits", payload["estimated_credits"]))
    body = "".join(f'<tr><th style="text-align:left">{e(k)}</th><td style="text-align:left">{e(v)}</td></tr>'
                   for k, v in rows)
    note = f"<p>{e(hint)}</p>" if hint else ""
    return f'<div class="welt-evaluation"><table><caption>Welt evaluation {e(payload["id"])}</caption>{body}</table>{note}</div>'


def _display(markup):
    try:
        from IPython.display import HTML, display
    except ImportError:
        raise OptionalDependencyError("show() needs IPython (Jupyter); use report_html() or report_link() instead.",
                                      code="optional_dependency_missing") from None
    display(HTML(markup))


# --- Waiting and fitting -------------------------------------------------------------

def _timeout(evaluation_id):
    display = safe_id(evaluation_id) or "evaluation_id"
    return JobTimeoutError(
        f"Waiting stopped; the evaluation continues. Reconnect with Client().evaluation('{display}').wait() "
        "using the same endpoint and current workspace credentials. Do not resubmit the evaluation.",
        code="job_wait_timeout", job_id=evaluation_id, retryable=True)


def _waiting(timeout, poll_interval):
    if timeout is None:
        _deadline(0, poll_interval)
        return None
    return _deadline(timeout, poll_interval)


def _settled(payload, api_key):
    """Raise for failed/cancelled; warn for partial results; True when finished."""
    if payload["status"] in ("failed", "cancelled"):
        fallback = (dict(code="job_cancelled", message="The evaluation was cancelled.") if payload["status"] == "cancelled"
                    else dict(code="execution_failed", message="The evaluation failed."))
        _failed(dict(payload, error=payload.get("error") or fallback), api_key)
    if payload["status"] == "partially_completed":
        missing = [m.get("name") or m.get("model_id") for m in payload["models"] if m.get("status") != "succeeded"]
        warn(EvaluationWarning(f"Evaluation {payload['id']} partially completed: {', '.join(map(str, missing))} did not "
                               "finish; the leaderboard covers the other models. See evaluation.models for errors."),
             stacklevel=3)
    return payload["status"] in FINISHED


def fit_request(payload, model_id):
    summary = _results(payload)
    ranked = [entry["model_id"] for entry in summary["leaderboard"]]
    if model_id is None and ranked:
        model_id = ranked[0]
    if model_id not in ranked:
        raise ValueError(f"model_id must be one of this evaluation's ranked models: {ranked}.")
    if payload.get("group_column") or payload.get("fold_column"):
        raise ValueError("This evaluation removed its split column from the features, but a fit on the stored "
                         "dataset would use it as a feature. Fit explicitly on a table without that column, "
                         "for example Classifier(model=...).fit(table.drop(columns=[...]), target=...).")
    evaluated = next(m for m in payload["models"] if m["model_id"] == model_id)
    return dict(dataset_id=payload["dataset_id"], model=model_id, task=payload["task"],
                configuration=evaluated.get("configuration") or "default", seed=payload["seed"]), evaluated


def fitted(predictor, evaluated, task, client):
    from .estimators import Classifier, Regressor
    if predictor.get("model_version") != evaluated.get("model_version"):
        warn(EvaluationWarning(
            f"{evaluated['model_id']} was evaluated as version {evaluated.get('model_version')} but fitted as "
            f"{predictor.get('model_version')}: the catalogue changed since the evaluation, so its scores describe "
            "a different version. Evaluate again to compare the current one."), stacklevel=3)
    estimator = Classifier if task == "classification" else Regressor
    return estimator._from_metadata(predictor, base_url=client.base_url, api_key=client.api_key)


# --- Handles -------------------------------------------------------------------------

class _EvaluationView:
    """Local state shared by Evaluation and AsyncEvaluation; no request is made here."""

    def __init__(self, client, payload):
        self.client, self._report, self._report_failed = client, None, False
        self._payload = _validated(payload, None, secret(client.api_key))
        self.id = self._payload["id"]

    def _update(self, payload):
        self._payload = _validated(payload, self.id, secret(self.client.api_key))
        self._report_failed = False  # A fresh read may make the report available again.
        return self

    def __repr__(self):
        return f"{type(self).__name__}(id={self.id!r}, status={self.status!r}, task={self.task!r})"

    def to_dict(self):
        """Return an independent copy of the last evaluation resource read from the server."""
        return deepcopy(self._payload)

    @property
    def status(self):
        return self._payload["status"]

    @property
    def stage(self):
        return self._payload.get("stage")

    @property
    def finished(self):
        return self.status in FINISHED

    @property
    def task(self):
        return self._payload["task"]

    @property
    def dataset_id(self):
        return self._payload.get("dataset_id")

    @property
    def target(self):
        return self._payload.get("target")

    @property
    def rows(self):
        return self._payload.get("rows")

    @property
    def primary_metric(self):
        return self._payload.get("primary_metric")

    @property
    def positive_class(self):
        return self._payload.get("positive_class")

    @property
    def classes(self):
        return deepcopy(self._payload.get("classes"))

    @property
    def models(self):
        return deepcopy(self._payload["models"])

    @property
    def skipped(self):
        return deepcopy(self._payload.get("skipped") or [])

    @property
    def estimated_credits(self):
        return self._payload.get("estimated_credits")

    @property
    def credits(self):
        """Credits settled at the outcome; None until the evaluation settles."""
        return self._payload.get("credits")

    @property
    def runtime(self):
        """Library versions the server used to compute the metrics."""
        return deepcopy(self._payload.get("runtime") or {})

    @property
    def comparison(self):
        return deepcopy((self._payload.get("summary") or {}).get("comparison"))

    @property
    def warnings(self):
        return deepcopy((self._payload.get("summary") or {}).get("warnings") or [])

    @property
    def predictions_expires_at(self):
        return self._payload.get("predictions_expires_at")

    def summary(self):
        """Leaderboard rows (models in rank order, then the baseline) with <metric>_mean/_sd columns."""
        return summary_frame(self._payload)

    def _status(self, hint=None):
        return status_html(self._payload, hint)


class Evaluation(_EvaluationView):
    """A cross-validated comparison; reads borrow the Client that created or reopened it."""

    def _path(self, suffix=""):
        return f"/v1/evaluations/{_id(self.id)}{suffix}"

    def refresh(self):
        return self._update(self.client.request("GET", self._path()))

    def wait(self, timeout=600, *, poll_interval=0.25):
        """Poll until finished; timeout=None waits without a local limit. Never cancels."""
        deadline = _waiting(timeout, poll_interval)
        api_key = secret(self.client.api_key)
        try:
            while True:
                self._update(self.client.request("GET", self._path(), _deadline=deadline, _job_id=self.id))
                if _settled(self._payload, api_key):
                    return self
                time.sleep(poll_interval if deadline is None else _wait(deadline, self.id, poll_interval))
                poll_interval = min(poll_interval * 1.4, 5)
        except JobTimeoutError:
            raise _timeout(self.id) from None

    def cancel(self):
        return self._update(self.client.request("POST", self._path("/cancel")))

    def delete(self):
        self.client.request("DELETE", self._path())
        self._report = None

    def folds(self):
        return folds_frame(self.client.request("GET", self._path("/folds")), self.id)

    def predictions(self):
        return predictions_frame(self.client.request("GET", self._path("/predictions")), self.id)

    def report_html(self):
        if self._report is None:
            self._report = self.client.request("GET", self._path("/report"), _text=True)
        return self._report

    def report_link(self, days=7):
        payload = self.client.request("POST", self._path("/report-links"), json=dict(expires_in_days=_days(days)))
        return report_link(payload, self.id)

    def report_links(self):
        return [report_link(item, self.id) for item in self.client.request("GET", self._path("/report-links"))["items"]]

    def revoke_report_link(self, link_id):
        self.client.request("DELETE", self._path(f"/report-links/{_id(_link_id(link_id))}"))

    def show(self):
        _display(notebook_html(self.report_html()) if self.finished else self._status())

    def _repr_html_(self):
        if self.finished and self._payload.get("report_available") and not self._report_failed:
            try:
                return notebook_html(self.report_html())
            except Exception:  # A notebook repr never raises; the status table remains useful.
                self._report_failed = True  # Not retried on every display until refresh().
        return self._status()

    def fit(self, model_id=None, *, timeout=600, idempotency_key=None):
        """Fit the chosen model on the full stored dataset: a new fit, not a promoted fold context."""
        request, evaluated = fit_request(self._payload, model_id)
        job = self.client.submit_fit(request.pop("dataset_id"), **request, idempotency_key=idempotency_key)
        return fitted(job.result(timeout=timeout), evaluated, self.task, self.client)


class AsyncEvaluation(_EvaluationView):
    """Async Evaluation; await its reads. It borrows the AsyncClient that created or reopened it."""

    def _path(self, suffix=""):
        return f"/v1/evaluations/{_id(self.id)}{suffix}"

    async def refresh(self):
        return self._update(await self.client.request("GET", self._path()))

    async def wait(self, timeout=600, *, poll_interval=0.25):
        """Await completion; timeout=None waits without a local limit. Never cancels."""
        deadline = _waiting(timeout, poll_interval)
        api_key = secret(self.client.api_key)
        try:
            while True:
                self._update(await self.client.request("GET", self._path(), _deadline=deadline, _job_id=self.id))
                if _settled(self._payload, api_key):
                    return self
                await asyncio.sleep(poll_interval if deadline is None else _wait(deadline, self.id, poll_interval))
                poll_interval = min(poll_interval * 1.4, 5)
        except JobTimeoutError:
            raise _timeout(self.id) from None

    async def cancel(self):
        return self._update(await self.client.request("POST", self._path("/cancel")))

    async def delete(self):
        await self.client.request("DELETE", self._path())
        self._report = None

    async def folds(self):
        return folds_frame(await self.client.request("GET", self._path("/folds")), self.id)

    async def predictions(self):
        return predictions_frame(await self.client.request("GET", self._path("/predictions")), self.id)

    async def report_html(self):
        if self._report is None:
            self._report = await self.client.request("GET", self._path("/report"), _text=True)
        return self._report

    async def report_link(self, days=7):
        payload = await self.client.request("POST", self._path("/report-links"),
                                            json=dict(expires_in_days=_days(days)))
        return report_link(payload, self.id)

    async def report_links(self):
        items = (await self.client.request("GET", self._path("/report-links")))["items"]
        return [report_link(item, self.id) for item in items]

    async def revoke_report_link(self, link_id):
        await self.client.request("DELETE", self._path(f"/report-links/{_id(_link_id(link_id))}"))

    async def show(self):
        _display(notebook_html(await self.report_html()) if self.finished else self._status())

    def _repr_html_(self):
        if self._report is not None:
            return notebook_html(self._report)
        return self._status("Run await evaluation.show() to display the report." if self.finished else None)

    async def fit(self, model_id=None, *, timeout=600, idempotency_key=None):
        """Fit the chosen model on the full stored dataset: a new fit, not a promoted fold context."""
        request, evaluated = fit_request(self._payload, model_id)
        job = await self.client.submit_fit(request.pop("dataset_id"), **request, idempotency_key=idempotency_key)
        return fitted(await job.result(timeout=timeout), evaluated, self.task, self.client)
