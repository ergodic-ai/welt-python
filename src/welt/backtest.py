"""Durable walk-forward backtests; handles borrow their sync or async Client."""

import asyncio
from copy import deepcopy
import math
import re
import time
from uuid import uuid4
import warnings

import numpy as np
import pandas as pd

from .client import _failed, _id, _wait
from .credentials import secret
from .errors import JobTimeoutError, TransportError, WeltError, safe_id
from .evaluation import _waiting
from .forecast import (_dataset_reference, _frequency, _integer, _levels, _names,
                       _univariate, _utc, _wire_frequency, _step, MAX_REQUEST_BYTES, NATIVE_QUANTILE_LEVELS)

TERMINAL = ("succeeded", "partially_completed", "failed", "cancelled")


class BacktestWarning(UserWarning):
    """Some requested backtest jobs did not complete successfully."""


def _table(value, label):
    dataset = _dataset_reference(value, label)
    if dataset:
        return dict(dataset_id=dataset)
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise ValueError(f"{label} needs a nonempty DataFrame or workspace dataset ID.")
    if value.columns.has_duplicates or any(not isinstance(c, str) or not c for c in value.columns):
        raise ValueError("Table columns must be unique nonempty strings.")
    def cell(v):
        if v is None or v is pd.NA or v is pd.NaT:
            return None
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, pd.Timestamp) and v.nanosecond:
            raise ValueError("Backtest timestamps require at most microsecond precision.")
        if isinstance(v, (pd.Timestamp,)) or hasattr(v, "isoformat"):
            return v.isoformat()
        if isinstance(v, float) and math.isnan(v):
            return None
        if type(v) not in (str, bool, int, float) or isinstance(v, float) and not math.isfinite(v):
            raise ValueError("Table cells must be finite JSON scalars, dates or missing values.")
        return v
    return dict(table=dict(columns=list(value.columns), rows=[[cell(v) for v in row]
                for row in value.itertuples(index=False, name=None)]))


def request(history, *, models, horizon, folds=5, step=None, context_length=None, seasonal_period=1,
            time=None, target=None, id=None, past_covariates=(), future_covariates=(), categorical=(),
            frequency=None, start=None, future=None, final_horizon=None,
            quantile_levels=(.1, .5, .9), cross_learning=False, primary_metric="mae", max_credits=None,
            expected_versions=None):
    """Build an owned JSON request. No uploads or prediction calls occur here."""
    if isinstance(models, str):
        raise ValueError("models must be a list of explicit model IDs.")
    models = list(models)
    if not 1 <= len(models) <= 8 or len(set(models)) != len(models) or any(safe_id(m) is None for m in models):
        raise ValueError("models must contain 1-8 unique explicit model IDs.")
    for value, low, high, label in ((horizon, 1, 1024, "horizon"), (folds, 1, 10, "folds"),
            (seasonal_period, 1, 100000, "seasonal_period")):
        _integer(value, low, high, label)
    for value, high, label in ((step, 100000, "step"), (context_length, 8192, "context_length"),
                              (final_horizon, 1024, "final_horizon"), (max_credits, None, "max_credits")):
        if value is not None:
            _integer(value, 1, high, label)
    if type(cross_learning) is not bool or primary_metric not in ("mae", "rmse", "mase", "pinball"):
        raise ValueError("Invalid cross_learning or primary_metric.")
    canon = _frequency(frequency) if frequency is not None else None
    if isinstance(history, pd.DataFrame):
        if start is not None:
            raise ValueError("start applies only to arrays without dates.")
        history = history.copy(deep=True)
        if time is None and isinstance(history.index, pd.DatetimeIndex):
            time = "__welt_time"
            while time in history.columns:
                time += "_"
            history.insert(0, time, history.index)
    elif _dataset_reference(history) is None:
        if time is not None or id is not None or past_covariates or future_covariates or categorical:
            raise ValueError("Arrays and Series carry one target; use a DataFrame for covariates or panels.")
        values = history.to_numpy(dtype=object) if isinstance(history, pd.Series) else np.asarray(history, dtype=object)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("Array history must be a nonempty one-dimensional sequence.")
        names = _names(target, "target")
        if len(names) > 1:
            raise ValueError("Arrays and Series carry exactly one target.")
        name = names[0] if names else "value"
        if isinstance(history, pd.Series) and isinstance(history.index, pd.DatetimeIndex):
            if start is not None:
                raise ValueError("start applies only to arrays without a datetime index.")
            time = "timestamp" if name != "timestamp" else "__welt_time"
            history = pd.DataFrame({time: history.index, name: values})
        elif start is not None:
            plan = _univariate(history, None, None, start, canon)
            time = plan["time"] if name != plan["time"] else "__welt_time"
            history = pd.DataFrame(plan["rows"], columns=[time, name])
        else:
            if canon is not None:
                raise ValueError("A frequency for an array requires start; omit both for a positional backtest.")
            history = pd.DataFrame({name: values})
        target = name
    elif start is not None:
        raise ValueError("A dataset's dates come from its time column; start applies only to arrays.")
    targets = _names(target, "target")
    past, known, categorical = (_names(v, name) for v, name in ((past_covariates, "past_covariates"),
                                 (future_covariates, "future_covariates"), (categorical, "categorical")))
    if not targets:
        raise ValueError("Name at least one target column.")
    roles = ([time] if time is not None else []) + ([id] if id is not None else []) + targets + past + known
    if (any(not isinstance(c, str) or not 1 <= len(c) <= 128 for c in roles)
            or len(set(roles)) != len(roles) or not set(categorical) <= set(past + known)):
        raise ValueError("Column roles must be distinct; categorical columns must be covariates.")
    if time is None and canon is not None:
        raise ValueError("Positional input has no calendar frequency; supply a time column.")
    if bool(final_horizon and known) != (future is not None):
        raise ValueError("future is required exactly for a final forecast with known-future covariates.")
    if isinstance(history, pd.DataFrame):
        if not set(roles) <= set(history.columns):
            raise ValueError("History lacks a declared column.")
        history = history.loc[:, roles]
        if time is not None:
            history = history.copy(deep=True)
            history[time] = [_utc(v, naive_utc=True).isoformat() for v in history[time]]
    body = dict(history=_table(history, "history"), models=models, time_column=time, id_column=id,
        targets=targets, past_covariates=past, future_covariates=known, categorical_covariates=categorical,
        horizon=int(horizon), folds=int(folds), seasonal_period=int(seasonal_period),
        quantile_levels=_levels(quantile_levels, NATIVE_QUANTILE_LEVELS), cross_learning=cross_learning,
        primary_metric=primary_metric)
    for name, value in (("step", step), ("context_length", context_length), ("final_horizon", final_horizon),
                        ("max_credits", max_credits)):
        if value is not None:
            body[name] = int(value)
    if canon is not None:
        body["frequency"] = _wire_frequency(canon)
    if future is not None:
        if isinstance(future, pd.DataFrame) and time is not None and time not in future.columns \
                and isinstance(future.index, pd.DatetimeIndex):
            future = future.copy(deep=True)
            future.insert(0, time, future.index)
        if isinstance(future, pd.DataFrame) and time is not None and time in future:
            future = future.copy(deep=True)
            future[time] = [_utc(v, naive_utc=True).isoformat() for v in future[time]]
        body["future"] = _table(future, "future")
    if expected_versions is not None:
        body["expected_versions"] = deepcopy(expected_versions)
    import json
    if len(json.dumps(body, allow_nan=False).encode()) > MAX_REQUEST_BYTES:
        raise ValueError("Backtest request exceeds 10 MiB; upload the history and use its dataset ID.")
    return body


def replay_key(value):
    if value is None:
        return uuid4().hex
    if not isinstance(value, str) or not 1 <= len(value) <= 128 or not value.isascii() or any(ord(c) < 33 for c in value):
        raise ValueError("idempotency_key must have 1-128 visible ASCII characters.")
    return value


def transport_error(key):
    return TransportError("Submission outcome is unknown. Retry the identical backtest with "
                          f"idempotency_key={key!r} to reconnect; do not create a new submission key.",
                          code="backtest_submission_unknown")


def timeout_error(backtest_id):
    return JobTimeoutError("Local wait expired; the backtest remains on the server. Reconnect with "
        f"Client(...).reopen_backtest({backtest_id!r}).wait() (await these calls for AsyncClient). "
        "Do not resubmit the backtest.", code="job_wait_timeout", job_id=backtest_id, retryable=True)


class _View:
    def __init__(self, client, payload, *, expected_id=None, expected_request=None):
        self.client = client
        self._owns_client = False
        self._payload = None
        self._update(payload, expected_id=expected_id)
        if expected_request is not None:
            try:
                request = expected_request
                if [m["model_id"] for m in payload["models"]] != request["models"]:
                    raise ValueError()
                for name in ("folds", "horizon", "context_length", "seasonal_period", "final_horizon",
                             "primary_metric", "targets", "quantile_levels"):
                    if payload.get(name) != request.get(name):
                        raise ValueError()
                if payload["step"] != request.get("step", request["horizon"]):
                    raise ValueError()
                if payload["index_kind"] != ("datetime" if request["time_column"] is not None else "position"):
                    raise ValueError()
                if request.get("frequency") and _frequency(request["frequency"]) != _frequency(payload["frequency"]):
                    raise ValueError()
                for model in payload["models"]:
                    pin = (request.get("expected_versions") or {}).get(model["model_id"])
                    if pin and (pin["model_version"] != model["model_version"]
                                or pin["configuration_version"] != model["configuration"]):
                        raise ValueError()
            except (KeyError, TypeError, ValueError):
                raise WeltError("Backtest response differs from the submitted request.", code="invalid_response") from None

    def _update(self, payload, expected_id=None):
        try:
            if (not isinstance(payload, dict) or safe_id(payload["id"]) is None
                    or not payload["id"].startswith("bt_") or payload["status"] not in ("queued", "running", *TERMINAL)
                    or payload["index_kind"] not in ("position", "datetime")
                    or expected_id is not None and payload["id"] != expected_id
                    or not isinstance(payload["models"], list) or not payload["models"]):
                raise ValueError()
            key = secret(self.client.api_key)
            if key and key in payload["id"]:
                raise ValueError()
            for name, low, high in (("folds", 1, 10), ("horizon", 1, 1024), ("step", 1, 100000),
                                    ("completed_jobs", 0, 88), ("total_jobs", 1, 88)):
                _integer(payload[name], low, high, name)
            if payload["completed_jobs"] > payload["total_jobs"]:
                raise ValueError()
            if payload["status"] in TERMINAL and payload["completed_jobs"] != payload["total_jobs"]:
                raise ValueError()
            if (payload["protocol_version"] != "backtest-protocol-v1"
                or payload["metrics_version"] != "backtest-metrics-v1"
                or payload["report_version"] != "backtest-report-v1"
                or not isinstance(payload["split_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", payload["split_sha256"]) is None):
                raise ValueError()
            if not 1 <= len(payload["models"]) <= 8:
                raise ValueError()
            _integer(payload["seasonal_period"], 1, 100000, "seasonal_period")
            if payload.get("context_length") is not None:
                _integer(payload["context_length"], 1, 8192, "context_length")
            if (not isinstance(payload["series_ids"], list) or not payload["series_ids"]
                or len(set(payload["series_ids"])) != len(payload["series_ids"])
                or any(s is not None and (not isinstance(s, str) or len(s) > 256) for s in payload["series_ids"])):
                raise ValueError()
            if not _names(payload["targets"], "targets") or payload["primary_metric"] not in ("mae", "rmse", "mase", "pinball"):
                raise ValueError()
            if payload.get("final_horizon") is not None:
                _integer(payload["final_horizon"], 1, 1024, "final_horizon")
            expected_folds = list(range(payload["folds"])) + ([None] if payload.get("final_horizon") else [])
            if payload["total_jobs"] != len(payload["models"]) * len(expected_folds):
                raise ValueError()
            _levels(payload["quantile_levels"], NATIVE_QUANTILE_LEVELS)
            if payload["index_kind"] == "datetime":
                _frequency(payload["frequency"])
            elif payload["frequency"] is not None:
                raise ValueError()
            ids, jobs, operations = set(), set(), set()
            for model in payload["models"]:
                if safe_id(model["model_id"]) is None or model["model_id"] in ids:
                    raise ValueError()
                ids.add(model["model_id"])
                if safe_id(model["model_version"]) is None or safe_id(model["configuration"]) is None:
                    raise ValueError()
                if [j["fold"] for j in model["jobs"]] != expected_folds:
                    raise ValueError()
                for job in model["jobs"]:
                    if (safe_id(job["job_id"]) is None or safe_id(job["operation_id"]) is None
                        or job["job_id"] in jobs or job["operation_id"] in operations):
                        raise ValueError()
                    jobs.add(job["job_id"])
                    operations.add(job["operation_id"])
            if self._payload is not None:
                for name in ("id", "index_kind", "models", "folds", "horizon", "step", "context_length",
                             "seasonal_period", "final_horizon", "primary_metric", "protocol_version", "split_sha256",
                             "frequency", "calendar_anchor", "targets", "series_ids", "quantile_levels",
                             "metrics_version", "report_version"):
                    if self._payload[name] != payload[name]:
                        raise ValueError()
        except (TypeError, KeyError, ValueError):
            raise WeltError("Invalid backtest response.", code="invalid_response") from None
        self._payload = deepcopy(payload)
        return self

    @property
    def id(self):
        return self._payload["id"]

    @property
    def status(self):
        return self._payload["status"]

    @property
    def finished(self):
        return self.status in TERMINAL

    @property
    def summary(self):
        return deepcopy(self._payload.get("summary"))

    def to_dict(self):
        return deepcopy(self._payload)

    def to_frame(self):
        """Pooled leaderboard, one row per completely backtested model; no remote call."""
        return pd.DataFrame([dict(model_id=r["model_id"], **r["values"]) for r in
                             (self.summary or {}).get("leaderboard", [])])

    def _path(self, suffix=""):
        return f"/v1/backtests/{_id(self.id)}{suffix}"

    def _settled(self):
        if self.status in ("failed", "cancelled"):
            _failed(dict(id=self.id, status=self.status, error={"code": "backtest_" + self.status,
                    "message": "Backtest " + self.status + "; inspect its metadata for job outcomes."}),
                    secret(self.client.api_key))
        if self.status == "partially_completed":
            warnings.warn("Backtest partially completed; inspect summary.jobs and incomplete_models.",
                          BacktestWarning, stacklevel=3)
        return self.finished

    def _predictions(self, payload, final):
        try:
            if payload["backtest_id"] != self.id or payload["index_kind"] != self._payload["index_kind"]:
                raise ValueError()
            rows = payload["final" if final else "items"]
            if final and self._payload["final_horizon"] is None and rows:
                raise ValueError()
            levels = _levels(payload["quantile_levels"], NATIVE_QUANTILE_LEVELS)
            if levels != self._payload["quantile_levels"]:
                raise ValueError()
            models = {m["model_id"] for m in self._payload["models"]} | {"baseline:naive", "baseline:seasonal_naive"}
            seen = set()
            for r in rows:
                key = (r["model_id"], r["fold"], r["id"], r["target"], r["lead"])
                if key in seen or r["model_id"] not in models:
                    raise ValueError()
                if r["target"] not in self._payload["targets"] or r["id"] not in self._payload["series_ids"]:
                    raise ValueError()
                seen.add(key)
                _integer(r["lead"], 1, self._payload["final_horizon"] if final else self._payload["horizon"], "lead")
                _integer(r["train_start"], 0, None, "train_start")
                _integer(r["train_stop"], 1, None, "train_stop")
                _integer(r["position"], 0, None, "position")
                if r["train_start"] >= r["train_stop"]:
                    raise ValueError()
                if final:
                    if r["fold"] is not None or r["actual"] is not None or r["model_id"].startswith("baseline:"):
                        raise ValueError()
                else:
                    _integer(r["fold"], 0, self._payload["folds"] - 1, "fold")
                if payload["index_kind"] == "position":
                    _integer(r["index"], 0, None, "index")
                    _integer(r["cutoff"], 0, None, "cutoff")
                    if r["index"] != r["cutoff"] + r["lead"] or r["index"] != r["position"]:
                        raise ValueError()
                elif _utc(r["index"]) != _step(_utc(r["cutoff"]), _frequency(self._payload["frequency"]),
                                               self._payload["calendar_anchor"], r["lead"]):
                    raise ValueError()
                if r["position"] != r["train_stop"] + r["lead"] - 1:
                    raise ValueError()
                for v in (r["actual"], r["prediction"]):
                    if v is not None and (type(v) not in (int, float) or not math.isfinite(v)):
                        raise ValueError()
                if r["quantiles"] is not None:
                    if r["model_id"].startswith("baseline:"):
                        raise ValueError()
                    if len(r["quantiles"]) != len(levels) or any(type(v) not in (int, float) or not math.isfinite(v)
                                                              for v in r["quantiles"]):
                        raise ValueError()
                    if (r["quantiles"] != sorted(r["quantiles"]) or r["prediction"] is None
                        or .5 in levels and r["prediction"] != r["quantiles"][levels.index(.5)]):
                        raise ValueError()
                elif not r["model_id"].startswith("baseline:"):
                    raise ValueError()
            return pd.DataFrame(deepcopy(rows))
        except (KeyError, TypeError, ValueError, OverflowError):
            raise WeltError("Invalid backtest prediction response.", code="invalid_response") from None


class Backtest(_View):
    def close(self):
        """Close only an estimator-created client; never cancel the server operation."""
        if self._owns_client:
            self.client.close()
            self._owns_client = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def refresh(self):
        return self._update(self.client.request("GET", self._path()))

    def wait(self, timeout=600, *, poll_interval=.25):
        """Wait locally; timeout never cancels or resubmits the durable backtest."""
        deadline = _waiting(timeout, poll_interval)
        try:
            while True:
                self._update(self.client.request("GET", self._path(), _deadline=deadline, _job_id=self.id))
                if self._settled():
                    return self
                time.sleep(poll_interval if deadline is None else _wait(deadline, self.id, poll_interval))
                poll_interval = min(poll_interval * 1.4, 5)
        except JobTimeoutError:
            raise timeout_error(self.id) from None

    def predictions(self):
        return self._predictions(self.client.request("GET", self._path("/predictions")), False)

    def final_predictions(self):
        return self._predictions(self.client.request("GET", self._path("/predictions")), True)

    def metrics(self):
        payload = self.client.request("GET", self._path("/metrics"))
        if payload.get("backtest_id") != self.id:
            raise WeltError("Invalid backtest metrics response.", code="invalid_response")
        return pd.DataFrame(deepcopy(payload["items"]))

    def report_html(self):
        return self.client.request("GET", self._path("/report"), _text=True)

    def cancel(self):
        return self._update(self.client.request("POST", self._path("/cancel")))

    def delete(self):
        self.client.request("DELETE", self._path())


class AsyncBacktest(_View):
    async def refresh(self):
        return self._update(await self.client.request("GET", self._path()))

    async def wait(self, timeout=600, *, poll_interval=.25):
        deadline = _waiting(timeout, poll_interval)
        try:
            while True:
                self._update(await self.client.request("GET", self._path(), _deadline=deadline, _job_id=self.id))
                if self._settled():
                    return self
                await asyncio.sleep(poll_interval if deadline is None else _wait(deadline, self.id, poll_interval))
                poll_interval = min(poll_interval * 1.4, 5)
        except JobTimeoutError:
            raise timeout_error(self.id) from None

    async def predictions(self):
        return self._predictions(await self.client.request("GET", self._path("/predictions")), False)

    async def final_predictions(self):
        return self._predictions(await self.client.request("GET", self._path("/predictions")), True)

    async def metrics(self):
        payload = await self.client.request("GET", self._path("/metrics"))
        if payload.get("backtest_id") != self.id:
            raise WeltError("Invalid backtest metrics response.", code="invalid_response")
        return pd.DataFrame(deepcopy(payload["items"]))

    async def report_html(self):
        return await self.client.request("GET", self._path("/report"), _text=True)

    async def cancel(self):
        return self._update(await self.client.request("POST", self._path("/cancel")))

    async def delete(self):
        await self.client.request("DELETE", self._path())
