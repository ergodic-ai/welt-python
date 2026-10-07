"""Remote estimators with scikit-learn construction, cloning and fitted-state behavior."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from .client import Client
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
        if any(not isinstance(c, str) for c in X.columns) or X.columns.has_duplicates:
            raise ValueError("DataFrame columns must be unique strings.")
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
            if pd.isna(value):
                value = None
            elif isinstance(value, np.generic):
                value = value.item()
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError("Table cells must be scalar numeric or categorical values.")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("Infinite values are unsupported.")
            values.append(value)
        rows.append(values)
    return columns, rows, named


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
        self.api_key = api_key
        self.random_state = random_state
        self.timeout = timeout

    def _client(self):
        return Client(base_url=self.base_url, api_key=self.api_key)

    def submit_fit(self, X, y):
        columns, rows, named = table(X)
        targets = np.asarray(y)
        if targets.ndim != 1 or len(targets) != len(rows):
            raise ValueError("y must be a one-dimensional target with one value per row.")
        if pd.isna(targets).any():
            raise ValueError("Targets must not contain missing values.")
        if self._task == "classification":
            check_classification_targets(targets)
        elif not np.issubdtype(targets.dtype, np.number):
            raise ValueError("Regression targets must be numeric.")
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

    def fit(self, X, y):
        columns, _, named = table(X)
        job = self.submit_fit(X, y)
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
        return estimator

    def predict_details(self, X, *, probabilities=False):
        check_is_fitted(self, "predictor_id_")
        columns, rows, named = table(X)
        if len(columns) != self.n_features_in_:
            raise ValueError("Prediction feature count differs from the fitted table.")
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
