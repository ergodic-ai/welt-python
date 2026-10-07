"""Declared official estimator checks; does not claim the complete estimator suite."""
import httpx
import pytest
from sklearn.utils import estimator_checks as checks
from welt import Classifier, Client, Regressor
from welt.estimators import _RemoteEstimator
from test_client import ServiceFixture


@pytest.mark.parametrize('estimator',[Classifier(),Regressor(),Classifier(api_key='invented-check-key')])
@pytest.mark.parametrize('check_name',[
    'check_estimator_cloneable', 'check_no_attributes_set_in_init',
    'check_parameters_default_constructible', 'check_get_params_invariance',
    'check_set_params',
])
def test_official_construction_parameters(estimator,check_name):
    getattr(checks,check_name)(type(estimator).__name__,estimator)


@pytest.mark.parametrize('estimator',[Classifier(),Regressor()])
@pytest.mark.parametrize('check_name',[
    'check_dict_unchanged', 'check_fit_check_is_fitted', 'check_fit_idempotent',
])
def test_official_supported_fitted_state(estimator,check_name,monkeypatch):
    transport=httpx.MockTransport(ServiceFixture())
    monkeypatch.setattr(_RemoteEstimator,'_client',lambda self:Client(api_key='fixture-only',transport=transport))
    getattr(checks,check_name)(type(estimator).__name__,estimator)
