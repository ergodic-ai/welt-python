"""Bounded zero-shot forecasting contexts and owned native forecast results."""
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import math
from types import MappingProxyType

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .client import Client, _deadline, _wait
from .credentials import credential, secret
from .errors import AuthenticationError, InvalidForecastResultError, WeltError, safe_id


def _integer(value, low, high, label):
    if isinstance(value, np.integer):
        value = value.item()
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer from {low} to {high}.")
    return value


def _numbers(values, *, observations=False):
    result = []
    for v in values:
        if isinstance(v, np.generic):
            v = v.item()
        if type(v) not in (int, float):
            raise ValueError("Forecast values must be finite numeric scalars.")
        try:
            valid = math.isfinite(v) and (not observations or abs(v) <= float(np.finfo(np.float32).max))
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("Forecast values must be finite; observations must fit FP32.")
        result.append(v)
    return result


def _utc(value, *, require_utc=False):
    try:
        t = pd.Timestamp(value)
        if pd.isna(t) or t.tzinfo is None or t.nanosecond:
            raise ValueError()
        if require_utc and t.utcoffset() != timedelta(0):
            raise ValueError()
        return t.tz_convert("UTC").to_pydatetime()
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Forecast timestamps need explicit timezone and microsecond precision.") from None


def _levels(levels):
    if isinstance(levels, (str, bytes)):
        raise ValueError("Quantiles must be an ordered sequence of native levels.")
    try:
        result = _numbers(list(levels))
    except TypeError:
        raise ValueError("Quantiles must be an ordered sequence of native levels.") from None
    if (not 1 <= len(result) <= 9 or result != sorted(set(result))
            or any(v not in [i / 10 for i in range(1, 10)] for v in result)):
        raise ValueError("Quantiles must be unique increasing native levels 0.1 through 0.9.")
    return result


def _future(series, horizon):
    start = _utc(series["start"], require_utc=True)
    try:
        return [start + timedelta(seconds=(len(series["values"]) + i)
                                   * series["frequency_seconds"]) for i in range(horizon)]
    except OverflowError:
        raise ValueError("Forecast time index exceeds the supported datetime range.") from None


def _series(history, time, target, start, frequency_seconds):
    times = None
    if isinstance(history, pd.DataFrame):
        if (not isinstance(time, str) or not isinstance(target, str) or time == target
                or history.columns.has_duplicates or time not in history or target not in history):
            raise ValueError("DataFrame forecasting requires distinct explicit time and target columns.")
        times = list(history[time])
        values = history[target].to_numpy(dtype=object)
    elif isinstance(history, pd.Series):
        if time is not None or target is not None:
            raise ValueError("time and target apply only to a DataFrame.")
        values = history.to_numpy(dtype=object)
        if isinstance(history.index, pd.DatetimeIndex):
            times = list(history.index)
    else:
        if time is not None or target is not None:
            raise ValueError("time and target apply only to a DataFrame.")
        values = np.asarray(history, dtype=object)
    if values.ndim != 1 or not 16 <= len(values) <= 512:
        raise ValueError("Forecast context must have 16 to 512 one-dimensional observations.")
    values = _numbers(values, observations=True)
    if times is not None:
        times = [_utc(t) for t in times]
        delta = times[1] - times[0]
        if delta.microseconds or delta.total_seconds() < 1:
            raise ValueError("Forecast observations need increasing whole-second spacing.")
        inferred = _integer(int(delta.total_seconds()), 1, 31_536_000, "Frequency")
        if any(b - a != delta for a, b in zip(times, times[1:])):
            raise ValueError("Forecast observations must be ordered, unique and regularly spaced.")
        if start is not None and _utc(start) != times[0]:
            raise ValueError("Explicit start differs from the observed time index.")
        if frequency_seconds is not None and _integer(frequency_seconds, 1, 31_536_000, "Frequency") != inferred:
            raise ValueError("Explicit frequency differs from the observed time index.")
        start, frequency_seconds = times[0], inferred
    else:
        if start is None or frequency_seconds is None:
            raise ValueError("Arrays or unindexed Series require start and frequency_seconds.")
        start = _utc(start)
        frequency_seconds = _integer(frequency_seconds, 1, 31_536_000, "Frequency")
    result = dict(values=values, start=start.isoformat(), frequency_seconds=frequency_seconds)
    _future(result, 64)
    return result


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value):
    if isinstance(value, MappingProxyType):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True, init=False)
class ForecastResult:
    """An immutable native result; conversions return independent ordinary copies."""
    _payload: object = field(repr=False)
    job_id: str

    def __init__(self, payload, *, job_id):
        try:
            p = deepcopy(payload)
            if not isinstance(p, dict) or set(p) != {
                    "task", "model_id", "model_version", "configuration", "context_length",
                    "horizon", "frequency_seconds", "timestamps", "predictions", "point_semantics",
                    "quantile_levels", "quantiles", "calibrated_intervals"} or safe_id(job_id) is None:
                raise ValueError()
            if p["task"] != "univariate_forecasting" or any(
                    safe_id(p[k]) is None for k in ("model_id", "model_version", "configuration")):
                raise ValueError()
            n = _integer(p["context_length"], 16, 512, "Context length")
            h = _integer(p["horizon"], 1, 64, "Horizon")
            freq = _integer(p["frequency_seconds"], 1, 31_536_000, "Frequency")
            if p["point_semantics"] != "median" or p["calibrated_intervals"] is not False:
                raise ValueError()
            levels = _levels(p["quantile_levels"])
            points = _numbers(p["predictions"])
            if any(not isinstance(t, str) for t in p["timestamps"]):
                raise ValueError()
            times = [_utc(t, require_utc=True) for t in p["timestamps"]]
            if len(points) != h or len(times) != h or len(p["quantiles"]) != h:
                raise ValueError()
            rows = [_numbers(row) for row in p["quantiles"]]
            if any(len(row) != len(levels) or row != sorted(row) for row in rows):
                raise ValueError()
            if any((b - a) != timedelta(seconds=freq) for a, b in zip(times, times[1:])):
                raise ValueError()
            if 0.5 in levels and any(not math.isclose(v, row[levels.index(0.5)], rel_tol=1e-6, abs_tol=1e-6)
                                     for v, row in zip(points, rows)):
                raise ValueError()
            # Store primitive owned values, never caller-owned arrays or dictionaries.
            p.update(context_length=n, horizon=h, frequency_seconds=freq,
                     predictions=points, quantile_levels=levels, quantiles=rows,
                     timestamps=list(p["timestamps"]))
        except (KeyError, TypeError, ValueError, OverflowError):
            raise InvalidForecastResultError("Invalid native forecast result; no repair was applied.",
                                             code="invalid_forecast_result") from None
        object.__setattr__(self, "_payload", _freeze(p))
        object.__setattr__(self, "job_id", job_id)

    def to_dict(self):
        """Return an independent JSON-compatible native payload; job_id is a separate attribute."""
        return _thaw(self._payload)

    @property
    def timestamps(self):
        return pd.DatetimeIndex(self._payload["timestamps"], name="timestamp")

    @property
    def predictions(self):
        return np.asarray(self._payload["predictions"], dtype=float).copy()

    @property
    def quantile_levels(self):
        return tuple(self._payload["quantile_levels"])

    @property
    def quantiles(self):
        return np.asarray(self._payload["quantiles"], dtype=float).copy()

    @property
    def model_id(self):
        return self._payload["model_id"]

    @property
    def model_version(self):
        return self._payload["model_version"]

    @property
    def configuration(self):
        return self._payload["configuration"]

    def to_frame(self):
        """Return future UTC rows with median and q_<level> columns; no remote call."""
        return pd.DataFrame(dict(median=self.predictions, **{
            f"q_{level:g}": self.quantiles[:, i] for i, level in enumerate(self.quantile_levels)
        }), index=self.timestamps)


class ForecastJob:
    """Typed durable run; estimator submissions own clients, reconnections borrow them."""
    def __init__(self, job, *, expected=None, owns_client=False):
        value = secret(job.client.api_key)
        if safe_id(job.id) is None or (value and value in job.id):
            raise WeltError("Invalid forecast job identity.", code="invalid_response")
        self._job, self._expected = job, deepcopy(expected)
        self._owns_client = owns_client
        self.id, self.client = job.id, job.client

    def __repr__(self):
        return f"ForecastJob(id={self.id!r})"

    def inspect(self):
        return self._validate_job(self._job.inspect())

    def _validate_job(self, job):
        try:
            if (job["id"] != self.id or job["operation"] != "model_run"
                    or job["task"] != "univariate_forecasting"
                    or any(safe_id(job[k]) is None for k in ("model_id", "model_version"))
                    or safe_id(job["resolved_configuration"]["version"]) is None):
                raise ValueError()
            if self._expected and (job["model_id"] != self._expected["payload"]["model_id"]
                    or job["model_version"] != self._expected["model_version"]
                    or job["resolved_configuration"]["version"] != self._expected["configuration"]):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise InvalidForecastResultError("Forecast job differs from the expected task or versions.",
                                             code="invalid_forecast_result", job_id=self.id) from None
        return job

    def cancel(self):
        self.inspect()
        return self._validate_job(self._job.cancel())

    def close(self):
        if self._owns_client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def result(self, *, timeout=600, poll_interval=0.25):
        deadline = _deadline(timeout, poll_interval)
        job = self._validate_job(self.client.request("GET", f"/v1/jobs/{self.id}",
                                _deadline=deadline, _job_id=self.id))
        raw = self._job.result(timeout=_wait(deadline, self.id, float("inf")), poll_interval=poll_interval)
        result = self._validate_result(raw, job)
        _wait(deadline, self.id, 0)
        return result

    def _validate_result(self, raw, job):
        result = ForecastResult(raw, job_id=self.id)
        try:
            if (result.model_id != job["model_id"] or result.model_version != job["model_version"]
                    or result.configuration != job["resolved_configuration"]["version"]
                    or ("row_count" in job and raw["context_length"] + raw["horizon"] != job["row_count"])):
                raise ValueError()
            if self._expected:
                p = self._expected["payload"]
                if (raw["context_length"] != len(p["series"]["values"])
                        or raw["horizon"] != p["horizon"]
                        or raw["frequency_seconds"] != p["series"]["frequency_seconds"]
                        or list(result.quantile_levels) != p["quantile_levels"]
                        or list(result.timestamps.to_pydatetime()) != _future(p["series"], p["horizon"])):
                    raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise InvalidForecastResultError("Forecast result differs from its submitted request.",
                                             code="invalid_forecast_result", job_id=self.id) from None
        return result


class AsyncForecastJob:
    """Async typed reconnection; existing AsyncClient submission retains its raw job contract."""
    def __init__(self, job):
        self._job, self._validator = job, ForecastJob(job)
        self.id, self.client = job.id, job.client

    def __repr__(self):
        return f"AsyncForecastJob(id={self.id!r})"

    async def inspect(self):
        return self._validator._validate_job(await self._job.inspect())

    async def cancel(self):
        await self.inspect()
        return self._validator._validate_job(await self._job.cancel())

    async def aclose(self):
        # Reconnected jobs borrow the enclosing AsyncClient.
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def result(self, *, timeout=600, poll_interval=0.25):
        deadline = _deadline(timeout, poll_interval)
        job = self._validator._validate_job(await self.client.request("GET", f"/v1/jobs/{self.id}",
                                           _deadline=deadline, _job_id=self.id))
        raw = await self._job.result(timeout=_wait(deadline, self.id, float("inf")), poll_interval=poll_interval)
        result = self._validator._validate_result(raw, job)
        _wait(deadline, self.id, 0)
        return result


class Forecast(BaseEstimator):
    """A zero-shot history context: fit snapshots history, prediction submits GPU work."""
    def __init__(self, model="chronos-2", configuration="default", base_url=None,
                 api_key=None, workspace=None, timeout=600):
        self.model, self.configuration, self.base_url = model, configuration, base_url
        self.api_key, self.workspace, self.timeout = credential(api_key), workspace, timeout

    def _client(self):
        return Client(base_url=self.base_url, api_key=self.api_key, workspace=self.workspace)

    def _clear_fitted(self):
        for name in list(vars(self)):
            if name.endswith("_") and not name.startswith("__"):
                delattr(self, name)

    def set_params(self, **params):
        if params:
            self._clear_fitted()
        if "api_key" in params:
            params["api_key"] = credential(params["api_key"])
        return super().set_params(**params)

    def _signature(self):
        return (self.model, self.configuration, self.base_url, self.workspace, self.timeout,
                self._digest(self.api_key))

    @staticmethod
    def _digest(key):
        return hashlib.sha256((secret(key) or "").encode()).hexdigest()

    def fit(self, history, *, time=None, target=None, start=None, frequency_seconds=None):
        """Copy context and resolve versions. No private observations are uploaded by fit."""
        self._clear_fitted()
        series = _series(history, time, target, start, frequency_seconds)
        if self.configuration != "default":
            raise ValueError("Initial forecasting supports configuration='default' only.")
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number.")
        with self._client() as client:
            workspace = client.workspace()
            meta = client.model(self.model)
            value = secret(client.api_key)
            if (not isinstance(meta, dict) or meta.get("id") != self.model
                    or not isinstance(meta.get("tasks"), list)
                    or "univariate_forecasting" not in meta["tasks"]
                    or not isinstance(meta.get("operations"), list)
                    or "model_run" not in meta["operations"]
                    or meta.get("supports_version_expectations") is not True):
                raise WeltError("Server/model does not support guarded univariate forecasting.",
                                code="unsupported_capability")
            try:
                model_version, config = meta["version"], meta["configuration"]["version"]
                if any(safe_id(v) is None or (value and value in v)
                       for v in (model_version, config, workspace["id"])):
                    raise ValueError()
                limits = meta["limits"]
                if not (_integer(limits["min_context"], 16, 512, "Minimum context") <= len(series["values"])
                        <= _integer(limits["max_context"], 16, 512, "Maximum context")):
                    raise ValueError()
                max_horizon = _integer(limits["max_horizon"], 1, 64, "Maximum horizon")
            except (KeyError, TypeError, ValueError):
                raise WeltError("Invalid forecasting metadata or unsupported context length.", code="invalid_response") from None
            self._origin_, self._key_digest_ = str(client.http.base_url), self._digest(client.api_key)
            self.workspace_id_ = workspace["id"]
        self._context_ = _freeze(series)
        self.model_version_, self.configuration_version_ = model_version, config
        self.context_length_, self.frequency_seconds_ = len(series["values"]), series["frequency_seconds"]
        self.cutoff_ = _future(series, 1)[0] - timedelta(seconds=series["frequency_seconds"])
        self._max_horizon_, self._parameters_ = max_horizon, self._signature()
        return self

    def submit_predict(self, horizon, *, quantile_levels=(0.1, 0.5, 0.9), idempotency_key=None):
        """Submit one run; retain job.id and explicitly close its client when finished."""
        check_is_fitted(self, "_context_")
        if self._signature() != self._parameters_:
            self._clear_fitted()
            raise ValueError("Forecast parameters changed; explicitly fit the context again.")
        horizon = _integer(horizon, 1, self._max_horizon_, "Horizon")
        levels = _levels(quantile_levels)
        payload = dict(model_id=self.model, task="univariate_forecasting",
                       series=_thaw(self._context_), horizon=horizon, quantile_levels=levels)
        client = self._client()
        try:
            if str(client.http.base_url) != self._origin_ or self._digest(client.api_key) != self._key_digest_:
                raise AuthenticationError("Forecast connection changed; explicitly fit again.", code="credential_mismatch")
            if client.workspace()["id"] != self.workspace_id_:
                raise AuthenticationError("Forecast workspace changed; explicitly fit again.", code="workspace_mismatch")
            job = client.submit_model_run(payload, idempotency_key=idempotency_key,
                         expected_model_version=self.model_version_,
                         expected_configuration_version=self.configuration_version_)
            return ForecastJob(job, owns_client=True, expected=dict(payload=payload, model_version=self.model_version_,
                                                  configuration=self.configuration_version_))
        except Exception:
            client.close()
            raise

    def predict_details(self, horizon, *, quantile_levels=(0.1, 0.5, 0.9), idempotency_key=None):
        """Submit one run, wait within timeout and return an owned ForecastResult."""
        with self.submit_predict(horizon, quantile_levels=quantile_levels,
                                 idempotency_key=idempotency_key) as job:
            return job.result(timeout=self.timeout)

    def predict(self, horizon, *, idempotency_key=None):
        """Return native median predictions as an array, shape (horizon,)."""
        return self.predict_details(horizon, idempotency_key=idempotency_key).predictions

    def predict_quantiles(self, horizon, *, quantile_levels=(0.1, 0.5, 0.9), idempotency_key=None):
        """Return native quantiles as an array, shape (horizon, n_quantiles)."""
        return self.predict_details(horizon, quantile_levels=quantile_levels,
                                    idempotency_key=idempotency_key).quantiles
