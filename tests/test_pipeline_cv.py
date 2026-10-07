"""API-048: fold isolation and scoring through the public sklearn interface.

The controlled service exercises client requests, not model qualification.
"""

import httpx
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from test_client import ServiceFixture
from welt import Classifier, Client, Regressor
from welt.estimators import _RemoteEstimator


@pytest.fixture
def service(monkeypatch):
    fixture = ServiceFixture()
    transport = httpx.MockTransport(fixture)
    monkeypatch.setattr(
        _RemoteEstimator, "_client",
        lambda self: Client(api_key="fixture-only", transport=transport),
    )
    return fixture


def assert_fold_upload(dataset, X, y, train):
    """Compare actual upload with an independently fitted fold transformation."""
    target_index = dataset["columns"].index(dataset["target"])
    uploaded_features = np.asarray([
        [value for index, value in enumerate(row) if index != target_index]
        for row in dataset["rows"]
    ])
    uploaded_targets = np.asarray([row[target_index] for row in dataset["rows"]])
    expected = StandardScaler().fit_transform(X[train])
    np.testing.assert_allclose(uploaded_features, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(uploaded_targets, y[train])
    assert len(dataset["rows"]) == len(train)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_cross_validation_uploads_only_independently_transformed_training_folds(service, task):
    X = np.column_stack((np.arange(18), np.arange(18) ** 2)).astype(float)
    if task == "classification":
        y = np.asarray([2, 7] * 9)
        splitter = StratifiedKFold(3, shuffle=True, random_state=42)
        estimator = Classifier(random_state=19)
        scoring = "accuracy"
    else:
        y = np.arange(18, dtype=float) ** 1.5
        splitter = KFold(3, shuffle=True, random_state=42)
        estimator = Regressor(random_state=19)
        scoring = "neg_mean_absolute_error"
    splits = list(splitter.split(X, y))
    pipeline = make_pipeline(StandardScaler(), estimator)
    original_parameters = estimator.get_params().copy()

    result = cross_validate(
        pipeline, X, y, cv=splits, scoring=scoring,
        return_estimator=True, error_score="raise", n_jobs=1,
    )

    assert len(service.datasets) == len(service.predictors) == 3
    assert len({model[-1].predictor_id_ for model in result["estimator"]}) == 3
    assert np.isfinite(result["test_score"]).all()
    assert estimator.get_params() == original_parameters
    with pytest.raises(NotFittedError):
        check_is_fitted(estimator)

    for dataset, fitted, (train, test), score in zip(
        service.datasets.values(), result["estimator"], splits, result["test_score"],
    ):
        assert_fold_upload(dataset, X, y, train)
        np.testing.assert_allclose(fitted[0].mean_, X[train].mean(axis=0))
        assert fitted[-1].get_params() == original_parameters
        assert fitted[-1].n_features_in_ == X.shape[1]
        assert fitted[-1].metadata()["seed"] == 19
        fresh = clone(fitted)
        with pytest.raises(NotFittedError):
            check_is_fitted(fresh[-1])
        assert not hasattr(fresh[-1], "predictor_id_")
        if task == "classification":
            np.testing.assert_array_equal(fitted[-1].classes_, [2, 7])
            probabilities = fitted.predict_proba(X[test])
            assert probabilities.shape == (len(test), 2)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1)
            np.testing.assert_array_equal(
                fitted[-1].classes_[probabilities.argmax(axis=1)], fitted.predict(X[test]),
            )
        else:
            # The controlled service returns its support-target mean. Verify
            # sklearn scoring consumed predictions from this fold's fit.
            expected_predictions = np.full(len(test), y[train].mean())
            np.testing.assert_allclose(fitted.predict(X[test]), expected_predictions)
            assert score == pytest.approx(-mean_absolute_error(y[test], expected_predictions))


def test_grid_search_clones_each_candidate_and_refits_only_the_selected_candidate(service):
    X = np.column_stack((np.arange(12), np.arange(12) ** 2)).astype(float)
    y = np.asarray([2, 7] * 6)
    splits = list(StratifiedKFold(2, shuffle=True, random_state=3).split(X, y))
    pipeline = make_pipeline(StandardScaler(), Classifier())
    search = GridSearchCV(
        pipeline, {"classifier__random_state": [4, 8]}, cv=splits,
        scoring="accuracy", n_jobs=1, error_score="raise", refit=True,
    ).fit(X, y)

    assert len(service.datasets) == len(service.predictors) == 5
    datasets = list(service.datasets.values())
    for candidate in range(2):
        for fold, (train, _) in enumerate(splits):
            assert_fold_upload(datasets[candidate * 2 + fold], X, y, train)
    assert_fold_upload(datasets[-1], X, y, np.arange(len(y)))
    seeds = [predictor["seed"] for predictor in service.predictors.values()]
    assert seeds[:4] == [4, 4, 8, 8]
    assert seeds[-1] == search.best_params_["classifier__random_state"]
    assert search.best_estimator_[-1].predictor_id_ == list(service.predictors)[-1]
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline[-1])
