"""Zero-shot forecasting contexts and owned native forecast results.

Two server contracts are served (DEC-085 §7/§8):

- ``forecasting`` (``POST /v1/forecasts``): long-format panels, multiple targets,
  past/future/categorical covariates and calendar frequencies, with limits from the
  model's published ``forecast_envelopes`` and ``min_context``.
- ``univariate_forecasting`` (hidden ``POST /v1/model-runs``): the 0.11–0.13 route,
  used only when the server does not advertise ``forecasting``.
"""
import calendar
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import json
import math
import re
from types import MappingProxyType

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .client import Client, _deadline, _id, _wait
from .credentials import credential, secret
from .errors import (AuthenticationError, InvalidForecastResultError, InvalidInputError,
                     TransportError, WeltError, safe_id)

#: Tables whose compact JSON encoding is at most this many bytes are sent inline in the
#: forecast request; a larger history or future table is uploaded first as a workspace
#: dataset (``Client.upload``) and referenced by dataset ID.
INLINE_TABLE_BYTES = 1024 * 1024

#: Native Chronos-2 quantile levels (DEC-085 §1), used when model metadata publishes none.
NATIVE_QUANTILE_LEVELS = tuple(v / 100 for v in (1, *range(5, 100, 5), 99))

#: Frequency aliases accepted by ``Forecast.fit(frequency=...)``, optionally prefixed by
#: a positive multiple such as ``"15min"`` or ``"2W"``. Month, quarter and year use
#: calendar arithmetic; whether a series is anchored on a day of the month (1–28) or on
#: month-ends comes from the observations, not from the alias (``M``, ``MS`` and ``ME``
#: all mean one calendar month).
FREQUENCY_ALIASES = MappingProxyType({
    "s": "second", "S": "second", "sec": "second", "second": "second",
    "min": "minute", "T": "minute", "minute": "minute",
    "h": "hour", "H": "hour", "hour": "hour",
    "D": "day", "d": "day", "day": "day",
    "W": "week", "week": "week",
    "M": "month", "MS": "month", "ME": "month", "month": "month",
    "Q": "quarter", "QS": "quarter", "QE": "quarter", "quarter": "quarter",
    "Y": "year", "YS": "year", "YE": "year", "A": "year", "AS": "year", "year": "year",
})
_SECONDS = {"week": 604_800, "day": 86_400, "hour": 3_600, "minute": 60, "second": 1}
_MONTHS = {"year": 12, "quarter": 3, "month": 1}
_LEGACY_LEVELS = [i / 10 for i in range(1, 10)]
_FP32 = float(np.finfo(np.float32).max)


def _integer(value, low, high, label):
    if isinstance(value, np.integer):
        value = value.item()
    if type(value) is not int or not low <= value <= (math.inf if high is None else high):
        bound = f"from {low} to {high}" if high is not None else f"of at least {low}"
        raise ValueError(f"{label} must be an integer {bound}.")
    return value


def _numbers(values, *, observations=False):
    result = []
    for v in values:
        if isinstance(v, np.generic):
            v = v.item()
        if type(v) not in (int, float):
            raise ValueError("Forecast values must be finite numeric scalars.")
        try:
            valid = math.isfinite(v) and (not observations or abs(v) <= _FP32)
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


def _levels(levels, allowed=None):
    """Validate requested quantile levels against an allowed set (legacy: 0.1 … 0.9)."""
    allowed = _LEGACY_LEVELS if allowed is None else list(allowed)
    if isinstance(levels, (str, bytes)):
        raise ValueError("Quantiles must be an ordered sequence of native levels.")
    try:
        result = _numbers(list(levels))
    except TypeError:
        raise ValueError("Quantiles must be an ordered sequence of native levels.") from None
    if (not 1 <= len(result) <= len(allowed) or result != sorted(set(result))
            or any(v not in allowed for v in result)):
        shown = ", ".join(f"{v:g}" for v in allowed)
        raise ValueError(f"Quantiles must be unique increasing native levels from: {shown}.")
    return result


def _future(series, horizon):
    start = _utc(series["start"], require_utc=True)
    try:
        return [start + timedelta(seconds=(len(series["values"]) + i)
                                   * series["frequency_seconds"]) for i in range(horizon)]
    except OverflowError:
        raise ValueError("Forecast time index exceeds the supported datetime range.") from None


# --- Frequencies and calendar arithmetic ---------------------------------------------

def _frequency(value):
    """Canonical (kind, size) for a user or wire frequency: ("s", seconds) or ("m", months)."""
    if isinstance(value, str):
        match = re.fullmatch(r"(\d{1,6})?([A-Za-z]+)", value)
        unit = FREQUENCY_ALIASES.get(match.group(2)) if match else None
        if unit is None:
            raise ValueError("Unsupported frequency alias. Use a {'unit', 'every'} dictionary or one of: "
                             + ", ".join(FREQUENCY_ALIASES) + " (optionally prefixed by a multiple, e.g. '15min').")
        every = int(match.group(1)) if match.group(1) else 1
    elif isinstance(value, (dict, MappingProxyType)):
        if set(value) != {"unit", "every"} or value["unit"] not in (*_SECONDS, *_MONTHS):
            raise ValueError("A frequency dictionary needs exactly unit (second, minute, hour, day, week, "
                             "month, quarter or year) and every.")
        unit, every = value["unit"], value["every"]
    else:
        raise ValueError("frequency must be a {'unit', 'every'} dictionary or an alias such as 'D', 'h' or 'MS'.")
    every = _integer(every, 1, None, "Frequency multiple")
    return ("s", every * _SECONDS[unit]) if unit in _SECONDS else ("m", every * _MONTHS[unit])


def _wire_frequency(canon):
    """The largest whole unit expressing a canonical frequency, as {unit, every}."""
    kind, size = canon
    for unit, step in (_SECONDS if kind == "s" else _MONTHS).items():
        if size % step == 0:
            return {"unit": unit, "every": size // step}
    raise AssertionError("unreachable")


def _anchor(times):
    """Calendar anchor: a common day of month (1–28) or 'end' when every time is a month-end."""
    days = {t.day for t in times}
    if len(days) == 1 and next(iter(days)) <= 28:
        return next(iter(days))
    if all(t.day == calendar.monthrange(t.year, t.month)[1] for t in times):
        return "end"
    return None


def _step(t, canon, anchor, k):
    kind, size = canon
    try:
        if kind == "s":
            return t + timedelta(seconds=size * k)
        year, month = divmod(t.year * 12 + t.month - 1 + size * k, 12)
        if not 1 <= year <= 9999:
            raise OverflowError()
        day = calendar.monthrange(year, month + 1)[1] if anchor == "end" else anchor
        return t.replace(year=year, month=month + 1, day=day)
    except OverflowError:
        raise ValueError("Forecast time index exceeds the supported datetime range.") from None


def _conforms(times, canon):
    """Return the anchor when sorted times follow canon without gaps, else raise ValueError."""
    if canon[0] == "s":
        anchor = None
    else:
        anchor = _anchor(times)
        if anchor is None or len({(t.hour, t.minute, t.second, t.microsecond) for t in times}) != 1:
            raise ValueError("Calendar frequencies need every observation on one day of the month (1–28) "
                             "or on month-ends, at one time of day.")
    if any(_step(a, canon, anchor, 1) != b for a, b in zip(times, times[1:])):
        raise ValueError("Forecast observations must be unique and regularly spaced at the frequency, "
                         "with no gaps. Welt does not fill, resample or truncate.")
    return anchor


def _infer(times):
    """Infer (canon, anchor) from at least two sorted, unique UTC times; ValueError if irregular."""
    deltas = {b - a for a, b in zip(times, times[1:])}
    if len(deltas) == 1:
        delta = next(iter(deltas))
        if delta.microseconds or delta.total_seconds() < 1:
            raise ValueError("Forecast observations need increasing whole-second spacing.")
        return ("s", int(delta.total_seconds())), None
    index = [t.year * 12 + t.month - 1 for t in times]
    months = {b - a for a, b in zip(index, index[1:])}
    if len(months) == 1 and next(iter(months)) >= 1:
        canon = ("m", next(iter(months)))
        try:
            return canon, _conforms(times, canon)
        except ValueError:
            pass
    raise ValueError("Forecast observations must be ordered, unique and regularly spaced (fixed seconds or "
                     "calendar months), with no gaps. Pass frequency= explicitly or prepare the series.")


# --- Table cells ----------------------------------------------------------------------

def _times(column):
    """UTC datetimes from an aware datetime column or aware timestamp cells; naive is rejected."""
    if isinstance(column.dtype, pd.DatetimeTZDtype):
        if column.isna().any() or (column.dt.nanosecond != 0).any():
            raise ValueError("Forecast timestamps need explicit timezone and microsecond precision.")
        return list(column.dt.tz_convert("UTC").dt.to_pydatetime())
    return [_utc(v) for v in column.tolist()]


def _missing(v):
    return v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and math.isnan(v))


def _numeric(column, label, *, missing):
    """JSON-ready numbers (None for missing when allowed): finite, FP32, never boolean."""
    dtype = column.dtype
    if isinstance(dtype, np.dtype) and dtype.kind in "iuf":
        values = column.to_numpy()
        nan = np.isnan(values) if dtype.kind == "f" else np.zeros(len(values), dtype=bool)
        if (nan.any() and not missing) or (dtype.kind == "f" and np.isinf(values).any()):
            raise ValueError(f"Column {label!r} must be finite" + ("" if missing else " and complete") + ".")
        if (np.abs(values[~nan].astype(float)) > _FP32).any():
            raise ValueError(f"Column {label!r} must fit FP32.")
        return [None if n else v for v, n in zip(values.tolist(), nan.tolist())]
    result = []
    for v in column.to_numpy(dtype=object):
        if isinstance(v, np.generic):
            v = v.item()
        if _missing(v):
            if not missing:
                raise ValueError(f"Column {label!r} must be complete.")
            result.append(None)
            continue
        if type(v) not in (int, float) or not math.isfinite(v) or abs(v) > _FP32:
            raise ValueError(f"Column {label!r} must contain finite FP32 numbers, not booleans or text.")
        result.append(v)
    return result


def _category(column, label):
    result = []
    for v in column.to_numpy(dtype=object):
        if isinstance(v, np.generic):
            v = v.item()
        if type(v) not in (str, int):
            raise ValueError(f"Categorical column {label!r} must contain complete string or integer cells.")
        result.append(v)
    return result


def _ids(column, label):
    result = []
    for v in column.to_numpy(dtype=object):
        if isinstance(v, np.generic):
            v = v.item()
        if type(v) not in (str, int) or (isinstance(v, str) and not v):
            raise ValueError(f"Series ID column {label!r} must contain complete nonempty string or integer cells.")
        result.append(v)
    return result


def _names(value, label):
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    try:
        names = list(value)
    except TypeError:
        raise ValueError(f"{label} must be a column name or a sequence of column names.") from None
    if any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError(f"{label} must be unique nonempty column names.")
    return names


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


# --- History parsing --------------------------------------------------------------------

def _resolve_frequency(frequency, frequency_seconds):
    if frequency is not None and frequency_seconds is not None:
        raise ValueError("Pass frequency or frequency_seconds, not both.")
    if frequency_seconds is not None:
        return ("s", _integer(frequency_seconds, 1, 31_536_000, "Frequency"))
    return None if frequency is None else _frequency(frequency)


def _univariate(history, time, target, start, canon):
    """The 0.11–0.13 single-series form: ordered, complete, finite FP32 history."""
    times = None
    if isinstance(history, pd.DataFrame):
        if (not isinstance(time, str) or not isinstance(target, str) or time == target
                or history.columns.has_duplicates or time not in history or target not in history):
            raise ValueError("DataFrame forecasting requires distinct explicit time and target columns.")
        times = _times(history[time])
        values = history[target].to_numpy(dtype=object)
        names = (time, target)
    elif isinstance(history, pd.Series):
        if time is not None or target is not None:
            raise ValueError("time and target apply only to a DataFrame.")
        values = history.to_numpy(dtype=object)
        if isinstance(history.index, pd.DatetimeIndex):
            times = _times(history.index.to_series())
        names = ("timestamp", "value")
    else:
        if time is not None or target is not None:
            raise ValueError("time and target apply only to a DataFrame.")
        values = np.asarray(history, dtype=object)
        names = ("timestamp", "value")
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Forecast history must be one-dimensional with at least two observations.")
    values = _numbers(values, observations=True)
    if times is not None:
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Forecast observations must be ordered, unique and regularly spaced.")
        if canon is None:
            canon, anchor = _infer(times)
        else:
            anchor = _conforms(times, canon)
        if start is not None and _utc(start) != times[0]:
            raise ValueError("Explicit start differs from the observed time index.")
    else:
        if start is None or canon is None:
            raise ValueError("Arrays or unindexed Series require start and frequency_seconds (or frequency).")
        first = _utc(start)
        anchor = None if canon[0] == "s" else _anchor([first])
        if anchor is None and canon[0] == "m":
            raise ValueError("A calendar start must fall on day 1–28 or a month-end.")
        times = [_step(first, canon, anchor, k) for k in range(len(values))]
    legacy = None
    if canon[0] == "s" and canon[1] <= 31_536_000:
        legacy = dict(values=values, start=times[0].isoformat(), frequency_seconds=canon[1])
    rows = [[t.isoformat(), v] for t, v in zip(times, values)]
    series = [dict(id=None, length=len(values), last=times[-1], anchor=anchor)]
    return dict(form="univariate", time=names[0], id=None, targets=[names[1]], past=[], future=[],
                categorical=[], columns=list(names), rows=rows, series=series, canon=canon,
                frequency=None, legacy=legacy)


def _long(history, time, target, id, past, future, categorical, canon):
    """DEC-085 long format: one row per (id, time); rows may be in any order."""
    if not isinstance(history, pd.DataFrame):
        raise ValueError("Panels, multiple targets and covariates need a long-format pandas DataFrame.")
    if not isinstance(time, str) or not time.strip():
        raise ValueError("Name the time column with time='column'.")
    targets = _names(target, "target")
    if not targets:
        raise ValueError("Name at least one target column with target='column' or a list.")
    past, future, categorical = (_names(past, "past_covariates"), _names(future, "future_covariates"),
                                 _names(categorical, "categorical"))
    if id is not None and (not isinstance(id, str) or not id.strip()):
        raise ValueError("id must be a column name.")
    roles = [time, *([id] if id is not None else []), *targets, *past, *future]
    if len(set(roles)) != len(roles):
        raise ValueError("time, id, target and covariate columns must all be distinct.")
    if history.columns.has_duplicates or any(name not in history for name in roles):
        missing = [name for name in roles if name not in history]
        raise ValueError(f"History columns must be unique and include every named column; missing: {missing[:10]}.")
    if not set(categorical) <= set(past) | set(future):
        raise ValueError("categorical must name past or future covariate columns.")
    if not len(history):
        raise ValueError("Forecast history must contain rows.")
    times = _times(history[time])
    ids = _ids(history[id], id) if id is not None else [None] * len(times)
    rendered = {}
    for raw in ids:
        if raw is not None and rendered.setdefault(str(raw), raw) != raw:
            raise ValueError(f"Series IDs {raw!r} and {rendered[str(raw)]!r} render alike; use distinct IDs.")
    columns = {}
    for name in targets:
        columns[name] = _numeric(history[name], name, missing=True)
    for name in past:
        columns[name] = (_category(history[name], name) if name in categorical
                         else _numeric(history[name], name, missing=True))
    for name in future:
        columns[name] = (_category(history[name], name) if name in categorical
                         else _numeric(history[name], name, missing=False))
    groups = {}
    for index, (raw, t) in enumerate(zip(ids, times)):
        groups.setdefault(raw, []).append((t, index))
    series, kinds = [], set()
    for raw, members in groups.items():
        members.sort(key=lambda m: m[0])
        stamps = [t for t, _ in members]
        label = "the series" if raw is None else f"series {str(raw)[:80]!r}"
        if any(b == a for a, b in zip(stamps, stamps[1:])):
            raise ValueError(f"Each (id, time) pair must appear once; {label} repeats a time.")
        if len(stamps) < 2:
            raise ValueError(f"History for {label} needs at least two observations.")
        try:
            if canon is None:
                found, anchor = _infer(stamps)
                kinds.add(found)
            else:
                anchor = _conforms(stamps, canon)
        except ValueError as error:
            raise ValueError(f"History for {label}: {error}") from None
        for name in targets:
            if all(columns[name][i] is None for _, i in members):
                raise ValueError(f"Target {name!r} is entirely missing for {label}.")
        series.append(dict(id=None if raw is None else str(raw), length=len(stamps), last=stamps[-1],
                           anchor=anchor))
    if canon is None:
        if len(kinds) != 1:
            raise ValueError("Every series must share one frequency; pass frequency= explicitly.")
        canon, frequency = kinds.pop(), None
    else:
        frequency = _wire_frequency(canon)
    wire = [time, *([id] if id is not None else []), *targets, *past, *future]
    data = [[t.isoformat() for t in times], *([ids] if id is not None else []),
            *[columns[name] for name in [*targets, *past, *future]]]
    return dict(form="long", time=time, id=id, targets=targets, past=past, future=future,
                categorical=categorical, columns=wire, rows=[list(r) for r in zip(*data)],
                series=series, canon=canon, frequency=frequency, legacy=None)


def _history(history, time, target, id, past, future, categorical, frequency, start, frequency_seconds):
    canon = _resolve_frequency(frequency, frequency_seconds)
    long = (id is not None or not isinstance(target, (str, type(None))) or any(
        _names(v, n) for v, n in ((past, "past_covariates"), (future, "future_covariates"),
                                   (categorical, "categorical"))))
    if long:
        if start is not None or frequency_seconds is not None:
            raise ValueError("start and frequency_seconds apply only to single-series arrays and Series; "
                             "use the time column and frequency= for long-format history.")
        return _long(history, time, target, id, past, future, categorical, canon)
    plan = _univariate(history, time, target, start, canon)
    if frequency is not None or plan["canon"][0] == "s":
        # A regular single series carries its exact spacing; the server need not infer it.
        plan["frequency"] = _wire_frequency(plan["canon"])
    return plan


def _future_table(plan, future, horizon):
    """Validate the known-future table: exactly horizon rows per series at its future times."""
    if not plan["future"]:
        if future is not None:
            raise ValueError("future applies only when the fit named future_covariates.")
        return None
    if not isinstance(future, pd.DataFrame):
        raise ValueError("Pass future=DataFrame with the time column, id column and every future covariate.")
    names = [plan["time"], *([plan["id"]] if plan["id"] is not None else []), *plan["future"]]
    if future.columns.has_duplicates or set(future.columns) != set(names):
        extra = sorted(str(c) for c in set(future.columns) - set(names))
        missing = sorted(set(names) - set(future.columns))
        raise ValueError(f"future must contain exactly the time, id and future covariate columns. "
                         f"Missing: {missing[:10]}. Extra: {extra[:10]}.")
    times = _times(future[plan["time"]])
    ids = _ids(future[plan["id"]], plan["id"]) if plan["id"] is not None else [None] * len(times)
    columns = {name: (_category(future[name], name) if name in plan["categorical"]
                      else _numeric(future[name], name, missing=False)) for name in plan["future"]}
    known = {s["id"]: s for s in plan["series"]}
    seen = {}
    for raw, t in zip(ids, times):
        key = None if raw is None else str(raw)
        if key not in known:
            raise ValueError(f"future contains series {str(raw)[:80]!r} that is not in the history.")
        seen.setdefault(key, []).append(t)
    for key, s in known.items():
        expected = [_step(s["last"], plan["canon"], s["anchor"], k) for k in range(1, horizon + 1)]
        if sorted(seen.get(key, [])) != expected:
            label = "the series" if key is None else f"series {key[:80]!r}"
            raise ValueError(f"future needs exactly {horizon} rows for {label}, at the {horizon} steps after "
                             "its last observation.")
    data = [[t.isoformat() for t in times], *([ids] if plan["id"] is not None else []),
            *[columns[name] for name in plan["future"]]]
    return dict(columns=names, rows=[list(r) for r in zip(*data)])


# --- Model metadata -------------------------------------------------------------------

def _view(meta, task):
    """Model metadata with a matching task profile's fields taking precedence."""
    profiles = meta.get("task_profiles")
    if isinstance(profiles, list):
        for profile in profiles:
            if isinstance(profile, dict) and profile.get("task") == task:
                return {**meta, **{k: v for k, v in profile.items() if k != "task"}}
    return meta


def _versions(view, value):
    model_version, config = view["version"], view["configuration"]["version"]
    if any(safe_id(v) is None or (value and value in v) for v in (model_version, config)):
        raise ValueError()
    return model_version, config


def _envelopes(view):
    limits = view.get("limits") if isinstance(view.get("limits"), dict) else {}
    corners = view.get("forecast_envelopes", limits.get("forecast_envelopes"))
    minimum = view.get("min_context", limits.get("min_context"))
    if not isinstance(corners, list) or not corners:
        raise ValueError()
    keys = ("max_context", "max_variates", "max_series", "max_horizon")
    checked = [{k: _integer(c[k], 1, None, k) for k in keys} for c in corners]
    levels = view.get("quantile_levels")
    levels = _numbers(list(NATIVE_QUANTILE_LEVELS) if levels is None else levels)
    if not levels or levels != sorted(set(levels)) or any(not 0 < v < 1 for v in levels):
        raise ValueError()
    return checked, _integer(minimum, 1, None, "Minimum context"), levels


def _capabilities(meta, model, value):
    """Resolve the forecasting route (preferred) or the legacy univariate route."""
    if not isinstance(meta, dict) or meta.get("id") != model:
        raise WeltError("Invalid forecasting metadata.", code="invalid_response")
    tasks = meta.get("tasks") if isinstance(meta.get("tasks"), list) else []
    if "forecasting" in tasks:
        view = _view(meta, "forecasting")
        operations = view.get("operations") if isinstance(view.get("operations"), list) else []
        if "forecast" in operations and view.get("supports_version_expectations") is True:
            try:
                model_version, config = _versions(view, value)
                corners, minimum, levels = _envelopes(view)
            except (KeyError, TypeError, ValueError):
                raise WeltError("Invalid forecasting metadata: version, configuration, forecast_envelopes "
                                "and min_context are required.", code="invalid_response") from None
            return dict(task="forecasting", model_version=model_version, configuration=config,
                        envelopes=corners, min_context=minimum, levels=levels)
    view = _view(meta, "univariate_forecasting")
    operations = view.get("operations") if isinstance(view.get("operations"), list) else []
    if ("univariate_forecasting" in tasks and "model_run" in operations
            and view.get("supports_version_expectations") is True):
        try:
            model_version, config = _versions(view, value)
            limits = view["limits"]
            bounds = (_integer(limits["min_context"], 16, 512, "Minimum context"),
                      _integer(limits["max_context"], 16, 512, "Maximum context"),
                      _integer(limits["max_horizon"], 1, 64, "Maximum horizon"))
        except (KeyError, TypeError, ValueError):
            raise WeltError("Invalid forecasting metadata.", code="invalid_response") from None
        return dict(task="univariate_forecasting", model_version=model_version, configuration=config,
                    min_context=bounds[0], max_context=bounds[1], max_horizon=bounds[2])
    raise WeltError("Server/model does not support guarded forecasting.", code="unsupported_capability")


def _corner(c):
    return (f"context ≤ {c['max_context']}, variates ≤ {c['max_variates']}, series ≤ {c['max_series']}, "
            f"horizon ≤ {c['max_horizon']}")


def _admit(envelopes, model, context, variates, series, horizon):
    """A request is admitted when some published corner dominates it (no interpolation)."""
    if not any(context <= c["max_context"] and variates <= c["max_variates"] and series <= c["max_series"]
               and horizon <= c["max_horizon"] for c in envelopes):
        raise InvalidInputError(
            f"No published {model} forecasting envelope admits context {context}, {variates} variate(s), "
            f"{series} series and horizon {horizon}. Published corners: "
            + "; ".join(_corner(c) for c in envelopes) + ". Shorten the history, reduce variates or series, "
            "or forecast a shorter horizon.", code="workload_limit")


def _metadata(client, model):
    """Catalogue entry, completed by the model detail route when guard fields are absent."""
    meta = client.model(model)
    if isinstance(meta, dict) and "supports_version_expectations" not in meta:
        detail = client.request("GET", f"/v1/models/{_id(model)}")
        if isinstance(detail, dict):
            meta = {**meta, **detail}
    return meta


# --- Results --------------------------------------------------------------------------

def _invalid(job_id=None, message="Invalid native forecast result; no repair was applied."):
    return InvalidForecastResultError(message, code="invalid_forecast_result", job_id=job_id)


def _quantile_rows(rows, levels, points, horizon):
    rows = [_numbers(row) for row in rows]
    if len(rows) != horizon or any(len(row) != len(levels) or row != sorted(row) for row in rows):
        raise ValueError()
    if 0.5 in levels and any(not math.isclose(v, row[levels.index(0.5)], rel_tol=1e-6, abs_tol=1e-6)
                             for v, row in zip(points, rows)):
        raise ValueError()
    return rows


def _forecasting_payload(p):
    if set(p) != {"task", "model_id", "model_version", "configuration", "horizon", "frequency",
                  "quantile_levels", "point_semantics", "calibrated_intervals", "cross_learning",
                  "rearranged_steps", "columns", "series"}:
        raise ValueError()
    if any(safe_id(p[k]) is None for k in ("model_id", "model_version", "configuration")):
        raise ValueError()
    h = _integer(p["horizon"], 1, None, "Horizon")
    canon = _frequency(p["frequency"])
    if (p["point_semantics"] != "median" or p["calibrated_intervals"] is not False
            or type(p["cross_learning"]) is not bool):
        raise ValueError()
    levels = _numbers(p["quantile_levels"])
    if not levels or levels != sorted(set(levels)) or any(not 0 < v < 1 for v in levels):
        raise ValueError()
    c = p["columns"]
    if not isinstance(c, dict) or set(c) != {"time", "id", "targets", "past_covariates",
                                             "future_covariates", "categorical_covariates"}:
        raise ValueError()
    if not isinstance(c["time"], str) or not (c["id"] is None or isinstance(c["id"], str)):
        raise ValueError()
    for key in ("targets", "past_covariates", "future_covariates", "categorical_covariates"):
        if not isinstance(c[key], list) or any(not isinstance(v, str) for v in c[key]):
            raise ValueError()
    if not c["targets"] or not isinstance(p["series"], list) or not p["series"]:
        raise ValueError()
    series, pairs = [], []
    for s in p["series"]:
        if not isinstance(s, dict) or set(s) != {"id", "target", "context_length", "timestamps",
                                                 "predictions", "quantiles"}:
            raise ValueError()
        if (c["id"] is None) != (s["id"] is None) or not (s["id"] is None or isinstance(s["id"], str)):
            raise ValueError()
        if s["target"] not in c["targets"]:
            raise ValueError()
        n = _integer(s["context_length"], 1, None, "Context length")
        if not isinstance(s["timestamps"], list) or any(not isinstance(t, str) for t in s["timestamps"]):
            raise ValueError()
        times = [_utc(t, require_utc=True) for t in s["timestamps"]]
        points = _numbers(s["predictions"])
        if len(times) != h or len(points) != h or not isinstance(s["quantiles"], list):
            raise ValueError()
        rows = _quantile_rows(s["quantiles"], levels, points, h)
        anchor = None if canon[0] == "s" else _anchor(times)
        if any(_step(a, canon, anchor, 1) != b for a, b in zip(times, times[1:])):
            raise ValueError()
        pairs.append((s["id"], s["target"]))
        series.append(dict(id=s["id"], target=s["target"], context_length=n,
                           timestamps=list(s["timestamps"]), predictions=points, quantiles=rows))
    # One entry per (id, target): grouped by id, in targets order within each id.
    width, order = len(c["targets"]), {t: i for i, t in enumerate(c["targets"])}
    if len(set(pairs)) != len(pairs) or len(pairs) % width:
        raise ValueError()
    blocks = [pairs[i:i + width] for i in range(0, len(pairs), width)]
    if (any(len({i for i, _ in b}) != 1 or [order[t] for _, t in b] != list(range(width)) for b in blocks)
            or len({b[0][0] for b in blocks}) != len(blocks)):
        raise ValueError()
    steps = _integer(p["rearranged_steps"], 0, len(series) * h, "Rearranged steps")
    p.update(horizon=h, quantile_levels=levels, series=series, rearranged_steps=steps,
             columns={k: (list(v) if isinstance(v, list) else v) for k, v in c.items()})
    return p


def _legacy_payload(p):
    if set(p) != {"task", "model_id", "model_version", "configuration", "context_length",
                  "horizon", "frequency_seconds", "timestamps", "predictions", "point_semantics",
                  "quantile_levels", "quantiles", "calibrated_intervals"}:
        raise ValueError()
    if any(safe_id(p[k]) is None for k in ("model_id", "model_version", "configuration")):
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
    rows = _quantile_rows(p["quantiles"], levels, points, h)
    if any((b - a) != timedelta(seconds=freq) for a, b in zip(times, times[1:])):
        raise ValueError()
    # Store primitive owned values, never caller-owned arrays or dictionaries.
    p.update(context_length=n, horizon=h, frequency_seconds=freq,
             predictions=points, quantile_levels=levels, quantiles=rows,
             timestamps=list(p["timestamps"]))
    return p


@dataclass(frozen=True, init=False)
class ForecastSeries:
    """One (series ID, target) forecast of a result; arrays are owned copies.

    id is the series ID rendered as a string (None without an id column); target is the
    target column (None for legacy univariate results).
    """
    id: object
    target: object
    context_length: int
    quantile_levels: tuple
    _timestamps: tuple = field(repr=False)
    _predictions: tuple = field(repr=False)
    _quantiles: tuple = field(repr=False)

    def __init__(self, *, id, target, context_length, quantile_levels, timestamps, predictions, quantiles):
        for name, value in (("id", id), ("target", target), ("context_length", context_length),
                            ("quantile_levels", tuple(quantile_levels)), ("_timestamps", tuple(timestamps)),
                            ("_predictions", tuple(predictions)),
                            ("_quantiles", tuple(tuple(row) for row in quantiles))):
            object.__setattr__(self, name, value)

    @property
    def timestamps(self):
        """Future UTC timestamps, one per horizon step."""
        return pd.DatetimeIndex(list(self._timestamps), name="timestamp")

    @property
    def predictions(self):
        """Native medians, shape (horizon,)."""
        return np.asarray(self._predictions, dtype=float).copy()

    @property
    def quantiles(self):
        """Native quantiles, shape (horizon, n_quantiles), in quantile_levels order."""
        return np.asarray(self._quantiles, dtype=float).reshape(len(self._predictions), -1).copy()


@dataclass(frozen=True, init=False)
class ForecastResult:
    """An immutable native result; conversions return independent ordinary copies.

    task is "forecasting" for POST /v1/forecasts results (one series entry per series ID
    and target; to_frame() is long format) or "univariate_forecasting" for legacy
    model-run results (to_frame() keeps the 0.11–0.13 UTC-indexed layout).
    """
    _payload: object = field(repr=False)
    job_id: str

    def __init__(self, payload, *, job_id):
        try:
            p = deepcopy(payload)
            if not isinstance(p, dict) or safe_id(job_id) is None:
                raise ValueError()
            if p.get("task") == "forecasting":
                p = _forecasting_payload(p)
            elif p.get("task") == "univariate_forecasting":
                p = _legacy_payload(p)
            else:
                raise ValueError()
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
            raise _invalid() from None
        object.__setattr__(self, "_payload", _freeze(p))
        object.__setattr__(self, "job_id", job_id)

    def to_dict(self):
        """Return an independent JSON-compatible native payload; job_id is a separate attribute."""
        return _thaw(self._payload)

    @property
    def task(self):
        return self._payload["task"]

    @property
    def series(self):
        """Tuple of ForecastSeries in result order (series ID first appearance, then targets order)."""
        p = self._payload
        if self.task == "univariate_forecasting":
            return (ForecastSeries(id=None, target=None, context_length=p["context_length"],
                                   quantile_levels=p["quantile_levels"], timestamps=p["timestamps"],
                                   predictions=p["predictions"], quantiles=p["quantiles"]),)
        return tuple(ForecastSeries(id=s["id"], target=s["target"], context_length=s["context_length"],
                                    quantile_levels=p["quantile_levels"], timestamps=s["timestamps"],
                                    predictions=s["predictions"], quantiles=s["quantiles"])
                     for s in p["series"])

    def _single(self):
        series = self.series
        if len(series) != 1:
            raise ValueError("This result has several series or targets; use .series or .to_frame().")
        return series[0]

    @property
    def timestamps(self):
        """Future UTC timestamps of a single-series, single-target result."""
        return self._single().timestamps

    @property
    def predictions(self):
        """Native medians, shape (horizon,), of a single-series, single-target result."""
        return self._single().predictions

    @property
    def quantiles(self):
        """Native quantiles, shape (horizon, n_quantiles), of a single-series, single-target result."""
        return self._single().quantiles

    @property
    def quantile_levels(self):
        return tuple(self._payload["quantile_levels"])

    @property
    def horizon(self):
        return self._payload["horizon"]

    @property
    def rearranged_steps(self):
        """Steps whose crossing native quantiles the server sorted; 0 for legacy results, which
        reject crossings instead."""
        return self._payload.get("rearranged_steps", 0)

    @property
    def frequency(self):
        """Native {unit, every} frequency of a forecasting result; None for legacy results."""
        value = self._payload.get("frequency")
        return None if value is None else _thaw(value)

    @property
    def cross_learning(self):
        return self._payload.get("cross_learning", False)

    @property
    def columns(self):
        """Echoed column roles of a forecasting result; None for legacy results."""
        value = self._payload.get("columns")
        return None if value is None else _thaw(value)

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
        """Return future UTC rows; no remote call.

        Forecasting results: long format with columns id, target, timestamp, median and
        q_<level>, one row per (series, target, step) in result order. Legacy results:
        UTC-indexed rows with median and q_<level> columns.
        """
        names = [f"q_{level:g}" for level in self.quantile_levels]
        if self.task == "univariate_forecasting":
            return pd.DataFrame(dict(median=self.predictions, **{
                name: self.quantiles[:, i] for i, name in enumerate(names)}), index=self.timestamps)
        parts = []
        for s in self.series:
            q, h = s.quantiles, len(s._predictions)
            parts.append(pd.DataFrame({"id": pd.Series([s.id] * h, dtype=object), "target": [s.target] * h,
                                       "timestamp": s.timestamps.rename(None), "median": s.predictions,
                                       **{name: q[:, i] for i, name in enumerate(names)}}))
        return pd.concat(parts, ignore_index=True)


# --- Jobs -----------------------------------------------------------------------------

_KINDS = {"univariate_forecasting": "model_run", "forecasting": "forecast"}


class ForecastJob:
    """Typed durable run; estimator submissions own clients, reconnections borrow them."""
    def __init__(self, job, *, expected=None, owns_client=False, datasets=None):
        value = secret(job.client.api_key)
        if safe_id(job.id) is None or (value and value in job.id):
            raise WeltError("Invalid forecast job identity.", code="invalid_response")
        self._job, self._expected = job, deepcopy(expected)
        self._owns_client = owns_client
        self.id, self.client = job.id, job.client
        # Workspace datasets this submission uploaded ({"history": id, "future": id}); empty when inline.
        self.datasets = dict(datasets or {})

    def __repr__(self):
        return f"ForecastJob(id={self.id!r})"

    def inspect(self):
        return self._validate_job(self._job.inspect())

    def _validate_job(self, job):
        try:
            kind = self._expected.get("kind", "univariate_forecasting") if self._expected else None
            if (job["id"] != self.id or job["task"] not in _KINDS or job["operation"] != _KINDS[job["task"]]
                    or (kind is not None and job["task"] != kind)
                    or any(safe_id(job[k]) is None for k in ("model_id", "model_version"))
                    or safe_id(job["resolved_configuration"]["version"]) is None):
                raise ValueError()
            if self._expected and (job["model_id"] != self._expected["model_id"]
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
            if (result.task != job["task"] or result.model_id != job["model_id"]
                    or result.model_version != job["model_version"]
                    or result.configuration != job["resolved_configuration"]["version"]):
                raise ValueError()
            if result.task == "univariate_forecasting":
                self._check_legacy(raw, job, result)
            elif self._expected:
                self._check_forecasting(result)
        except (KeyError, TypeError, ValueError):
            raise _invalid(self.id, "Forecast result differs from its submitted request.") from None
        return result

    def _check_legacy(self, raw, job, result):
        if "row_count" in job and raw["context_length"] + raw["horizon"] != job["row_count"]:
            raise ValueError()
        if self._expected:
            p = self._expected["payload"]
            if (raw["context_length"] != len(p["series"]["values"])
                    or raw["horizon"] != p["horizon"]
                    or raw["frequency_seconds"] != p["series"]["frequency_seconds"]
                    or list(result.quantile_levels) != p["quantile_levels"]
                    or list(result.timestamps.to_pydatetime()) != _future(p["series"], p["horizon"])):
                raise ValueError()

    def _check_forecasting(self, result):
        e, got = self._expected["request"], result.columns
        want = e["columns"]
        if (result.horizon != e["horizon"] or list(result.quantile_levels) != e["quantile_levels"]
                or result.cross_learning != e["cross_learning"]
                or _frequency(result.frequency) != tuple(e["canon"])
                or any(got[k] != want[k] for k in ("time", "id", "targets", "past_covariates", "future_covariates"))
                or sorted(got["categorical_covariates"]) != sorted(want["categorical_covariates"])):
            raise ValueError()
        series = result.series
        if len(series) != len(e["series"]):
            raise ValueError()
        for s, (sid, target, length, times) in zip(series, e["series"]):
            if (s.id != sid or s.target != target or s.context_length != length
                    or list(s.timestamps.to_pydatetime()) != list(times)):
                raise ValueError()


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


# --- Estimator ------------------------------------------------------------------------

def _encoded(table):
    """Bytes of the compact UTF-8 JSON encoding httpx sends."""
    return len(json.dumps(table, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


class Forecast(BaseEstimator):
    """A zero-shot history context: fit snapshots history, forecasting submits GPU work.

    Servers advertising the forecasting task receive POST /v1/forecasts requests (panels,
    multiple targets, covariates; limits from published envelopes). Older servers keep the
    guarded univariate model-run route for one regular series.
    """
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

    def fit(self, history, *, time=None, target=None, id=None, past_covariates=(), future_covariates=(),
            categorical=(), frequency=None, start=None, frequency_seconds=None):
        """Copy and validate history, then resolve the route, versions and limits. Uploads nothing.

        history is a long-format DataFrame (one row per series ID and time), a Series with a
        DatetimeIndex, or an array with start and frequency_seconds. target is a column name
        or a list; id names the series column (absent: one series). frequency is a
        {unit, every} dictionary or an alias (FREQUENCY_ALIASES); without it the spacing is
        checked locally and the server infers it.
        """
        self._clear_fitted()
        plan = _history(history, time, target, id, past_covariates, future_covariates, categorical,
                        frequency, start, frequency_seconds)
        if self.configuration != "default":
            raise ValueError("Forecasting supports configuration='default' only.")
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number.")
        with self._client() as client:
            workspace = client.workspace()
            value = secret(client.api_key)
            route = _capabilities(_metadata(client, self.model), self.model, value)
            if (not isinstance(workspace, dict) or safe_id(workspace.get("id")) is None
                    or (value and value in workspace["id"])):
                raise WeltError("Invalid workspace metadata.", code="invalid_response")
            lengths = [s["length"] for s in plan["series"]]
            variates = len(plan["targets"]) + len(plan["past"]) + len(plan["future"])
            if route["task"] == "forecasting":
                if min(lengths) < route["min_context"]:
                    raise InvalidInputError(f"Each series needs at least {route['min_context']} observations "
                                            f"(min_context of {self.model}).", code="workload_limit")
                _admit(route["envelopes"], self.model, max(lengths), variates, len(lengths), 1)
            else:
                if plan["legacy"] is None:
                    raise WeltError("This server forecasts only one regular series with fixed spacing up to one "
                                    "year (univariate_forecasting); panels, covariates and calendar frequencies "
                                    "need a server with the forecasting task.", code="unsupported_capability")
                if not route["min_context"] <= lengths[0] <= route["max_context"]:
                    raise InvalidInputError(f"Forecast context must have {route['min_context']} to "
                                            f"{route['max_context']} observations.", code="workload_limit")
                _future(plan["legacy"], route["max_horizon"])
            self._origin_, self._key_digest_ = str(client.http.base_url), self._digest(client.api_key)
            self.workspace_id_ = workspace["id"]
        self._route_ = _freeze(route)
        self._plan_ = _freeze({k: v for k, v in plan.items() if k != "legacy"})
        if route["task"] != "forecasting":
            self._context_ = _freeze(plan["legacy"])
        self.task_ = route["task"]
        self.model_version_, self.configuration_version_ = route["model_version"], route["configuration"]
        self.context_length_ = max(lengths)
        self.n_series_ = len(lengths)
        self.series_ids_ = tuple(s["id"] for s in plan["series"])
        self.targets_ = tuple(plan["targets"])
        self.frequency_ = _wire_frequency(plan["canon"])
        self.frequency_seconds_ = plan["canon"][1] if plan["canon"][0] == "s" else None
        self.cutoff_ = max(s["last"] for s in plan["series"])
        self._max_horizon_ = (route["max_horizon"] if route["task"] != "forecasting" else
                              max(c["max_horizon"] for c in route["envelopes"]))
        self._parameters_ = self._signature()
        return self

    # -- submission ---------------------------------------------------------------------

    def _ready(self):
        check_is_fitted(self, "_plan_")
        if self._signature() != self._parameters_:
            self._clear_fitted()
            raise ValueError("Forecast parameters changed; explicitly fit the context again.")

    def _connected(self):
        client = self._client()
        try:
            if str(client.http.base_url) != self._origin_ or self._digest(client.api_key) != self._key_digest_:
                raise AuthenticationError("Forecast connection changed; explicitly fit again.", code="credential_mismatch")
            if client.workspace()["id"] != self.workspace_id_:
                raise AuthenticationError("Forecast workspace changed; explicitly fit again.", code="workspace_mismatch")
        except Exception:
            client.close()
            raise
        return client

    def _submit_legacy(self, horizon, quantile_levels, idempotency_key):
        horizon = _integer(horizon, 1, self._max_horizon_, "Horizon")
        levels = _levels(quantile_levels)
        payload = dict(model_id=self.model, task="univariate_forecasting",
                       series=_thaw(self._context_), horizon=horizon, quantile_levels=levels)
        client = self._connected()
        try:
            job = client.submit_model_run(payload, idempotency_key=idempotency_key,
                         expected_model_version=self.model_version_,
                         expected_configuration_version=self.configuration_version_)
            return ForecastJob(job, owns_client=True, expected=dict(
                kind="univariate_forecasting", model_id=self.model, payload=payload,
                model_version=self.model_version_, configuration=self.configuration_version_))
        except Exception:
            client.close()
            raise

    def _submit(self, horizon, future, quantile_levels, cross_learning, idempotency_key):
        route, plan = _thaw(self._route_), _thaw(self._plan_)
        horizon = _integer(horizon, 1, None, "Horizon")
        levels = _levels(quantile_levels, route["levels"])
        if type(cross_learning) is not bool:
            raise ValueError("cross_learning must be True or False.")
        if idempotency_key is not None and safe_id(idempotency_key) is None:
            raise ValueError("idempotency_key must be a bounded opaque identifier.")
        variates = len(plan["targets"]) + len(plan["past"]) + len(plan["future"])
        _admit(route["envelopes"], self.model, self.context_length_, variates, self.n_series_, horizon)
        future_table = _future_table(plan, future, horizon)
        canon = tuple(plan["canon"])
        expected_series = []
        for s in plan["series"]:
            times = [_step(s["last"], canon, s["anchor"], k) for k in range(1, horizon + 1)]
            expected_series += [(s["id"], t, s["length"], times) for t in plan["targets"]]
        tables = dict(history=dict(columns=plan["columns"], rows=plan["rows"]))
        if future_table is not None:
            tables["future"] = future_table
        body = dict(model_id=self.model, configuration=self.configuration, time_column=plan["time"],
                    targets=plan["targets"], past_covariates=plan["past"], future_covariates=plan["future"],
                    categorical_covariates=plan["categorical"], horizon=horizon, quantile_levels=levels,
                    cross_learning=cross_learning)
        if plan["id"] is not None:
            body["id_column"] = plan["id"]
        if plan["frequency"] is not None:
            body["frequency"] = plan["frequency"]
        expected = dict(kind="forecasting", model_id=self.model, model_version=self.model_version_,
                        configuration=self.configuration_version_, request=dict(
                            horizon=horizon, quantile_levels=levels, cross_learning=cross_learning, canon=canon,
                            columns=dict(time=plan["time"], id=plan["id"], targets=plan["targets"],
                                         past_covariates=plan["past"], future_covariates=plan["future"],
                                         categorical_covariates=plan["categorical"]),
                            series=expected_series))
        client = self._connected()
        datasets = {}
        try:
            for name, table in tables.items():
                if _encoded(table) <= INLINE_TABLE_BYTES:
                    body[name] = dict(table=table)
                    continue
                dataset = client.upload(columns=table["columns"], rows=table["rows"], name=f"Forecast {name}")
                if not isinstance(dataset, dict) or safe_id(dataset.get("id")) is None:
                    raise WeltError("Invalid dataset response.", code="invalid_response")
                datasets[name] = dataset["id"]
                body[name] = dict(dataset_id=dataset["id"])
            try:
                job = client.submit_forecast(body, idempotency_key=idempotency_key,
                                             expected_model_version=self.model_version_,
                                             expected_configuration_version=self.configuration_version_)
            except TransportError:
                if not datasets:
                    raise
                raise TransportError(
                    "Could not reach Welt API while submitting the forecast; it may have been accepted. Uploaded "
                    f"datasets {datasets} remain. With an idempotency_key, replay Client.submit_forecast with the "
                    "same body referencing these dataset IDs and the same key instead of submitting again.",
                    code="transport_error", retryable=True) from None
            return ForecastJob(job, owns_client=True, expected=expected, datasets=datasets)
        except Exception:
            client.close()
            raise

    def submit_forecast(self, horizon, *, future=None, quantile_levels=(0.1, 0.5, 0.9),
                        cross_learning=False, idempotency_key=None):
        """Submit one guarded POST /v1/forecasts request without waiting; retain job.id.

        future is the known-future DataFrame (time, id and future covariate columns, exactly
        horizon rows per series), required exactly when the fit named future_covariates.
        Tables above INLINE_TABLE_BYTES are uploaded as workspace datasets first (job.datasets).
        """
        self._ready()
        if self.task_ != "forecasting":
            raise WeltError("forecast() needs a server with the forecasting task (POST /v1/forecasts); "
                            "this server offers only univariate predict().", code="unsupported_capability")
        return self._submit(horizon, future, quantile_levels, cross_learning, idempotency_key)

    def forecast(self, horizon, *, future=None, quantile_levels=(0.1, 0.5, 0.9), cross_learning=False,
                 idempotency_key=None):
        """Submit one forecast, wait within timeout and return an owned ForecastResult."""
        with self.submit_forecast(horizon, future=future, quantile_levels=quantile_levels,
                                  cross_learning=cross_learning, idempotency_key=idempotency_key) as job:
            return job.result(timeout=self.timeout)

    def submit_predict(self, horizon, *, future=None, quantile_levels=(0.1, 0.5, 0.9), idempotency_key=None):
        """Submit one single-series, single-target run without waiting; retain job.id."""
        self._ready()
        if self.n_series_ != 1 or len(self.targets_) != 1:
            raise ValueError("predict() returns one series and one target; this fit has "
                             f"{self.n_series_} series and {len(self.targets_)} target(s). Use forecast().")
        if self.task_ != "forecasting":
            if future is not None:
                raise ValueError("future applies only when the fit named future_covariates.")
            return self._submit_legacy(horizon, quantile_levels, idempotency_key)
        return self._submit(horizon, future, quantile_levels, False, idempotency_key)

    def predict_details(self, horizon, *, future=None, quantile_levels=(0.1, 0.5, 0.9), idempotency_key=None):
        """Submit one single-output run, wait within timeout and return an owned ForecastResult."""
        with self.submit_predict(horizon, future=future, quantile_levels=quantile_levels,
                                 idempotency_key=idempotency_key) as job:
            return job.result(timeout=self.timeout)

    def predict(self, horizon, future=None, *, idempotency_key=None):
        """Return native median predictions as an array, shape (horizon,), for one series and target."""
        return self.predict_details(horizon, future=future, idempotency_key=idempotency_key).predictions

    def predict_quantiles(self, horizon, *, future=None, quantile_levels=(0.1, 0.5, 0.9), idempotency_key=None):
        """Return native quantiles as an array, shape (horizon, n_quantiles), for one series and target."""
        return self.predict_details(horizon, future=future, quantile_levels=quantile_levels,
                                    idempotency_key=idempotency_key).quantiles
