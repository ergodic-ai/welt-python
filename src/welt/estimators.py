"""Remote estimators with scikit-learn construction, cloning and fitted-state behavior."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from .client import Client
from .credentials import credential
from .errors import PredictionPendingError


def table(X):
    if isinstance(X, (str, Path)):
        path = Path(X)
        X = (
            pd.read_parquet(path)
            if path.suffix.lower() in (".parquet", ".pq")
            else pd.read_csv(path)
        )
    if isinstance(X, pd.DataFrame):
        if any(not isinstance(c, str) or not c.strip() for c in X.columns) or X.columns.has_duplicates:
            raise ValueError("DataFrame columns must be unique, nonempty strings.")
        columns = list(X.columns)
        array = X.to_numpy(dtype=object)
        named = True
    else:
        array = np.asarray(X, dtype=object)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional table.")
        columns = [f"x{i}" for i in range(array.shape[1])]
        named = False
    if not array.shape[0] or not array.shape[1]:
        raise ValueError("X must contain rows and features.")
    rows = []
    for row in array:
        values = []
        for value in row:
            if isinstance(value, np.generic):
                value = value.item()
            if value is pd.NA or value is pd.NaT:
                value = None
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError("Table cells must be scalar numeric or categorical values.")
            if pd.isna(value):
                value = None
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("Infinite values are unsupported.")
            values.append(value)
        rows.append(values)
    return columns, rows, named


def _training_table(X, y, target):
    if (y is None) == (target is None):
        raise ValueError("Provide exactly one of y or target='column_name'.")
    if target is not None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("target requires a pandas DataFrame; use fit(X, y) for arrays or files.")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a nonempty column name.")
        if any(not isinstance(c, str) or not c.strip() for c in X.columns) or X.columns.has_duplicates:
            raise ValueError("DataFrame columns must be unique, nonempty strings.")
        if target not in X.columns:
            raise ValueError("The target column is missing; provide its exact DataFrame column name.")
        y = X[target]
        X = X.drop(columns=[target])
    columns, rows, named = table(X)
    targets = np.asarray(y)
    if targets.ndim != 1 or len(targets) != len(rows):
        raise ValueError("y must be a one-dimensional target with one value per row.")
    if pd.isna(targets).any():
        raise ValueError("Targets must not contain missing values.")
    return columns, rows, named, targets


class _RemoteEstimator(BaseEstimator):
    def __init__(
        self,
        model="tabicl-v2",
        configuration="default",
        base_url=None,
        api_key=None,
        random_state=0,
        timeout=600,
    ):
        self.model = model
        self.configuration = configuration
        self.base_url = base_url
        self.api_key = credential(api_key)
        self.random_state = random_state
        self.timeout = timeout

    def _client(self):
        return Client(base_url=self.base_url, api_key=self.api_key)

    def set_params(self, **params):
        if "api_key" in params:
            params["api_key"] = credential(params["api_key"])
        return super().set_params(**params)

    def _clear_fitted(self):
        for name in list(vars(self)):
            if name.endswith("_") and not name.startswith("__"):
                delattr(self, name)

    def submit_fit(self, X, y=None, *, target=None):
        """Submit preparation; specify y or a DataFrame target column explicitly."""
        columns, rows, _, targets = _training_table(X, y, target)
        return self._submit_table(columns, rows, targets)

    def _submit_table(self, columns, rows, targets):
        if self._task == "classification":
            check_classification_targets(targets)
        elif not np.issubdtype(targets.dtype, np.number):
            raise ValueError("Regression targets must be numeric.")
        elif not np.isfinite(targets).all():
            raise ValueError("Regression targets must be finite.")
        target_name = "__welt_target__"
        while target_name in columns:
            target_name += "_"
        target_values = [v.item() if isinstance(v, np.generic) else v for v in targets]
        client = self._client()
        try:
            dataset = client.upload(
                columns=columns + [target_name],
                rows=[r + [v] for r, v in zip(rows, target_values)],
                target=target_name,
            )
            job = client.submit_fit(
                dataset["id"],
                model=self.model,
                task=self._task,
                configuration=self.configuration,
                seed=self.random_state,
            )
        except Exception:
            client.close()
            raise
        return job

    def fit(self, X, y=None, *, target=None):
        """Fit with separate labels or fit(dataframe, target="column_name")."""
        # A failed refit cannot leave an old remote identity looking newly fitted.
        self._clear_fitted()
        columns, rows, named, targets = _training_table(X, y, target)
        job = self._submit_table(columns, rows, targets)
        try:
            predictor = job.result(timeout=self.timeout)
        finally:
            job.client.close()
        self._set_fitted(predictor)
        if named:
            self.feature_names_in_ = np.asarray(columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        return self

    def _set_fitted(self, predictor):
        if predictor["task"] != self._task:
            raise ValueError("Predictor task does not match this estimator.")
        self.predictor_id_ = predictor["id"]
        self.model_version_ = predictor["model_version"]
        self.n_features_in_ = len(predictor["columns"])
        self._columns_ = predictor["columns"]
        if self._task == "classification":
            self.classes_ = np.asarray(predictor["classes"])
        return self

    @classmethod
    def from_predictor(cls, predictor_id, *, base_url=None, api_key=None):
        with Client(base_url=base_url, api_key=api_key) as client:
            predictor = client.predictor(predictor_id)
        estimator = cls(
            model=predictor["model_id"],
            configuration=predictor["configuration"],
            base_url=base_url,
            api_key=api_key,
            random_state=predictor["seed"],
        )
        estimator._set_fitted(predictor)
        estimator.feature_names_in_ = np.asarray(predictor["columns"], dtype=object)
        return estimator

    def predict_details(self, X, *, probabilities=False):
        check_is_fitted(self, "predictor_id_")
        columns, rows, named = table(X)
        if named and set(columns) != set(self._columns_):
            missing = sorted(set(self._columns_) - set(columns))
            extra = sorted(set(columns) - set(self._columns_))
            # Bound schema context; no rows, values, or full input representations.
            def names(values):
                return repr([name[:80] for name in values[:10]]) + (" (more omitted)" if len(values) > 10 else "")
            raise ValueError(f"Prediction names differ from the fitted table. Missing: {names(missing)}. "
                             f"Extra: {names(extra)}. Supply only the fitted feature columns.")
        if len(columns) != self.n_features_in_:
            raise ValueError(f"X has {len(columns)} features, but {type(self).__name__} "
                             f"is expecting {self.n_features_in_} features as input.")
        if not named:
            columns = self._columns_
        with self._client() as client:
            try:
                return client.request(
                    "POST",
                    f"/v1/predictors/{self.predictor_id_}/predict",
                    json=dict(columns=columns, rows=rows, probabilities=probabilities),
                )
            except PredictionPendingError as pending:
                if pending.job_id is None:
                    raise
                return client.job(pending.job_id).result(timeout=self.timeout)

    def predict(self, X):
        return np.asarray(self.predict_details(X)["predictions"])

    def metadata(self):
        """Inspect pinned predictor metadata through current server authorization."""
        check_is_fitted(self, "predictor_id_")
        with self._client() as client:
            return client.predictor(self.predictor_id_)

    def export_metadata(self):
        """Return JSON-compatible server metadata; no key or executable context."""
        return self.metadata()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.categorical = True
        tags.non_deterministic = True
        return tags


class Classifier(ClassifierMixin, _RemoteEstimator):
    _task = "classification"

    def predict_proba(self, X):
        result = self.predict_details(X, probabilities=True)
        if result["probabilities"] is None:
            raise ValueError("This model did not return classification probabilities.")
        return np.asarray(result["probabilities"])


class Regressor(RegressorMixin, _RemoteEstimator):
    _task = "regression"
