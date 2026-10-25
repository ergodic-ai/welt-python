"""Additive CPU benchmark contracts; native results remain SDK0.15 compatible."""

from copy import deepcopy
import json
import math

from .errors import WeltError
from .forecast import _frequency, _step, _utc, _integer

IDS = {
    "benchmark:" + n
    for n in ("drift", "auto-theta", "auto-ets", "mfles", "ridge", "elastic-net", "laplace")
}
TERMINAL = ("succeeded", "partially_completed", "timed_out", "failed", "cancelled")


def validate(value, resource):
    try:
        if (
            value["backtest_id"] != resource["id"]
            or value["suite_version"] != "fast-v1"
            or value.get("split_sha256") not in (None, resource["split_sha256"])
            or value["index_kind"] != resource["index_kind"]
            or value["quantile_levels"] != resource["quantile_levels"]
            or value["expires_at"] != resource["predictions_expires_at"]
        ):
            raise ValueError()
        meta = value["metadata"]
        if (
            meta["suite_version"] != "fast-v1"
            or meta["status"] not in TERMINAL
            or meta["budget_seconds"] != 20
            or len(meta["models"]) != len(IDS)
            or {m["model_id"] for m in meta["models"]} != IDS
        ):
            raise ValueError()
        units = {}
        for u in value["units"]:
            fold, series, target = json.loads(u["key"])
            key = (u["model_id"], fold, series, target)
            if (
                key in units
                or key[0] not in IDS
                or series not in resource["series_ids"]
                or target not in resource["targets"]
                or (fold is None and resource["final_horizon"] is None)
                or (
                    fold is not None
                    and (type(fold) is not int or not 0 <= fold < resource["folds"])
                )
                or u["status"] not in ("succeeded", "failed", "skipped", "timed_out")
            ):
                raise ValueError()
            units[key] = u["status"]
        count = (
            len(IDS)
            * len(resource["series_ids"])
            * len(resource["targets"])
            * (resource["folds"] + bool(resource["final_horizon"]))
        )
        if len(units) != count and not (not units and meta["status"] in ("failed", "cancelled")):
            raise ValueError()
        seen, windows, actuals, origins = set(), {}, {}, {}
        common = None
        for final, rows in ((False, value["items"]), (True, value["final"])):
            for row in rows:
                key = (row["model_id"], row["fold"], row["id"], row["target"])
                h, lead = resource["final_horizon"] if final else resource["horizon"], row["lead"]
                if (
                    units.get(key) != "succeeded"
                    or (key[1] is None) != final
                    or type(lead) is not int
                    or not 1 <= lead <= h
                    or (key, lead) in seen
                    or row["position"] != row["train_stop"] + lead - 1
                    or not 0 <= row["train_start"] < row["train_stop"]
                ):
                    raise ValueError()
                for name in ("train_start", "train_stop", "position"):
                    _integer(row[name], 1 if name == "train_stop" else 0, None, name)
                first = row["train_stop"] - (
                    (resource["folds"] - 1) * resource["step"] + resource["horizon"]
                    if final
                    else key[1] * resource["step"]
                )
                if first < 1:
                    raise ValueError()
                if common is None:
                    common = first, row["train_stop"], row["cutoff"]
                if first != common[0]:
                    raise ValueError()
                if resource["index_kind"] == "datetime" and _utc(row["cutoff"]) != _step(
                    _utc(common[2]),
                    _frequency(resource["frequency"]),
                    resource["calendar_anchor"],
                    row["train_stop"] - common[1],
                ):
                    raise ValueError()
                if (
                    resource["context_length"] is not None
                    and row["train_start"] != row["train_stop"] - resource["context_length"]
                ):
                    raise ValueError()
                if resource["context_length"] is None and row["train_start"] != 0:
                    raise ValueError()
                seen.add((key, lead))
                windows.setdefault(key, set()).add(lead)
                if final and row["actual"] is not None:
                    raise ValueError()
                if resource["index_kind"] == "position":
                    _integer(row["index"], 0, None, "index")
                    _integer(row["cutoff"], 0, None, "cutoff")
                    if row["index"] != row["position"] or row["cutoff"] != row["train_stop"] - 1:
                        raise ValueError()
                elif _utc(row["index"]) != _step(
                    _utc(row["cutoff"]),
                    _frequency(resource["frequency"]),
                    resource["calendar_anchor"],
                    lead,
                ):
                    raise ValueError()
                for v in (row["actual"], row["prediction"]):
                    if v is not None and (type(v) not in (int, float) or not math.isfinite(v)):
                        raise ValueError()
                if row["prediction"] is None:
                    raise ValueError()
                q = row["quantiles"]
                levels = value["quantile_levels"]
                if key[0] == "benchmark:laplace":
                    if (
                        not isinstance(q, list)
                        or len(q) != len(levels)
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in q)
                        or q != sorted(q)
                        or 0.5 in levels
                        and row["prediction"] != q[levels.index(0.5)]
                    ):
                        raise ValueError()
                elif q is not None:
                    raise ValueError()
                if not final:
                    actualkey = (key[2], key[3], row["position"])
                    if actualkey in actuals and actuals[actualkey] != row["actual"]:
                        raise ValueError()
                    actuals[actualkey] = row["actual"]
        expected = {key for key, status in units.items() if status == "succeeded"}
        if set(windows) != expected or any(
            len(leads) != (resource["final_horizon"] if key[1] is None else resource["horizon"])
            for key, leads in windows.items()
        ):
            raise ValueError()
        eligible = {
            model
            for model in IDS
            if all(
                units.get((model, fold, series, target)) == "succeeded"
                for fold in range(resource["folds"])
                for series in resource["series_ids"]
                for target in resource["targets"]
            )
        }
        if {r["model_id"] for r in value["leaderboard"]} != eligible:
            raise ValueError()
        for r in value["metrics"] + value["leaderboard"]:
            if r["model_id"] not in IDS or any(
                v is not None and (type(v) not in (int, float) or not math.isfinite(v))
                for v in r["values"].values()
            ):
                raise ValueError()
        if len(value["items"]) + len(value["final"]) > 10000:
            raise ValueError()
        validate_extra(value, resource, units)
        return deepcopy(value)
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        raise WeltError("Invalid backtest benchmark response.", code="invalid_response") from None


def features(value):
    rows = []
    for d in value["details"]:
        fold, series, target = json.loads(d["key"])
        for lead in d.get("leads", []):
            for feature in lead["selected_features"]:
                rows.append(
                    dict(
                        model_id=d["model_id"],
                        fold=fold,
                        id=series,
                        target=target,
                        lead=lead["lead"],
                        alpha=lead["alpha"],
                        feature=feature["name"],
                        coefficient=feature["coefficient"],
                    )
                )
    return rows


def validate_extra(value, resource, units):
    meta = value["metadata"]
    allowed = {
        "pending",
        "running",
        "succeeded",
        "partially_completed",
        "timed_out",
        "failed",
        "skipped",
        "cancelled",
    }
    total = (
        len(resource["series_ids"])
        * len(resource["targets"])
        * (resource["folds"] + bool(resource["final_horizon"]))
    )
    for model in meta["models"]:
        name = model["model_id"].split(":", 1)[1]
        version = (
            "skaters-0.16.0"
            if name == "laplace"
            else "statsforecast-2.1.1"
            if name in ("auto-theta", "auto-ets", "mfles")
            else "sklearn-1.9.1"
            if name in ("ridge", "elastic-net")
            else "welt-drift-v1"
        )
        if (
            model["model_version"] != version
            or model["configuration"] != "fast-v1:" + name
            or model["status"] not in allowed
        ):
            raise ValueError()
        if units:
            completed = sum(
                status == "succeeded"
                for key, status in units.items()
                if key[0] == model["model_id"]
            )
            if (
                model["completed_units"] != completed
                or model["total_units"] != total
                or (model["status"] == "succeeded") != (completed == total)
            ):
                raise ValueError()
        for expected in (resource.get("benchmarks") or {}).get("models", []):
            if expected["model_id"] == model["model_id"]:
                for key in ("model_version", "configuration", "point_semantics", "covariates"):
                    if expected[key] != model[key]:
                        raise ValueError()
    groups, overall = set(), {}
    for row in value["metrics"]:
        dimension = row["dimension"]
        if dimension not in ("overall", "fold", "series", "lead"):
            raise ValueError()
        if dimension == "fold":
            _integer(row["fold"], 0, resource["folds"] - 1, "fold")
        if dimension == "lead":
            _integer(row["lead"], 1, resource["horizon"], "lead")
        if dimension == "series" and (
            row["id"] not in resource["series_ids"] or row["target"] not in resource["targets"]
        ):
            raise ValueError()
        key = (
            dimension,
            row["model_id"],
            row.get("fold"),
            row.get("id"),
            row.get("target"),
            row.get("lead"),
        )
        if key in groups:
            raise ValueError()
        groups.add(key)
        _integer(row["prediction_events"], 0, 10000, "prediction_events")
        _integer(row["observed_events"], 0, row["prediction_events"], "observed_events")
        for name in ("mae", "rmse", "mase", "pinball", "coverage", "interval_width"):
            v = row["values"][name]
            count = row["counts"][name]
            _integer(count, 0, row["observed_events"], "count")
            if v is None:
                if count != 0 or not isinstance(row["undefined"][name], str):
                    raise ValueError()
            elif (
                type(v) not in (float, int)
                or not math.isfinite(v)
                or v < 0
                or count == 0
                or (name == "coverage" and v > 1)
            ):
                raise ValueError()
            if (
                row["model_id"] != "benchmark:laplace"
                and name in ("pinball", "coverage", "interval_width")
                and (v is not None or count != 0)
            ):
                raise ValueError()
        if dimension == "overall":
            overall[row["model_id"]] = {k: v for k, v in row.items() if k != "dimension"}
    leaderboard = value["leaderboard"]
    if len({r["model_id"] for r in leaderboard}) != len(leaderboard) or any(
        overall.get(r["model_id"]) != r for r in leaderboard
    ):
        raise ValueError()
    detailkeys = set()
    for detail in value["details"]:
        fold, series, target = json.loads(detail["key"])
        model = detail["model_id"]
        key = (model, fold, series, target)
        if key in detailkeys or units.get(key) != "succeeded":
            raise ValueError()
        detailkeys.add(key)
        leads = detail.get("leads", [])
        if leads and model not in ("benchmark:ridge", "benchmark:elastic-net"):
            raise ValueError()
        seen = set()
        for item in leads:
            lead = item["lead"]
            _integer(
                lead, 1, resource["final_horizon"] if fold is None else resource["horizon"], "lead"
            )
            if lead in seen:
                raise ValueError()
            seen.add(lead)
            if item["alpha"] not in (
                (0.01, 0.1, 1.0) if model == "benchmark:elastic-net" else (1.0, 10.0, 100.0)
            ):
                raise ValueError()
            if item["l1_ratio"] != (0.5 if model == "benchmark:elastic-net" else 0.0):
                raise ValueError()
            _integer(item["training_rows"], 12, 4096, "training_rows")
            _integer(item["feature_count"], 1, 160, "feature_count")
            features = item["selected_features"]
            names = [f["name"] for f in features]
            if (
                len(names) > item["feature_count"]
                or len(set(names)) != len(names)
                or any(not isinstance(n, str) or not 1 <= len(n) <= 512 for n in names)
            ):
                raise ValueError()
            if any(
                type(f["coefficient"]) not in (int, float) or not math.isfinite(f["coefficient"])
                for f in features
            ):
                raise ValueError()
