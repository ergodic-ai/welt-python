"""Credential/fitted-state invariants, independent of model quality or API source."""
import pickle

import httpx
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from welt import Classifier, Client, Credential, Regressor
from welt.estimators import _RemoteEstimator, table
from test_client import ServiceFixture


def test_constructor_clone_nested_repr_html_and_pickle_reauthenticate(monkeypatch):
    secret='invented-credential-never-display'
    model=Classifier(api_key=secret)
    assert model.api_key.get_secret_value()==secret
    assert clone(model).api_key.get_secret_value()==secret
    assert clone(model).get_params()==model.get_params()
    pipeline=make_pipeline(model)
    for value in (repr(model),repr(model.get_params()),repr(pipeline),pipeline._repr_html_()):
        assert secret not in value
    payload=pickle.dumps(model)
    assert secret.encode() not in payload
    restored=pickle.loads(payload)
    assert restored.api_key.get_secret_value() is None
    monkeypatch.setenv('WELT_API_KEY','invented-new-session-key')
    with restored._client() as client:
        assert client.http.headers['Authorization']=='Bearer invented-new-session-key'
    model.set_params(api_key='another-invented-key')
    assert isinstance(model.api_key,Credential) and 'another-invented-key' not in repr(model)


def test_failed_refit_clears_remote_identity_and_feature_metadata(monkeypatch):
    service=ServiceFixture();transport=httpx.MockTransport(service)
    monkeypatch.setattr(_RemoteEstimator,'_client',lambda self:Client(api_key='fixture-only',transport=transport))
    model=Classifier().fit([[0],[1]],[0,1])
    prior=model.predictor_id_
    with pytest.raises(ValueError):model.fit([[0]],[np.nan])
    with pytest.raises(NotFittedError):model.predict([[0]])
    assert prior in service.predictors
    assert not hasattr(model,'classes_') and not hasattr(model,'n_features_in_')


def test_nonfinite_targets_and_nonscalar_cells_reject_before_http(monkeypatch):
    monkeypatch.setattr(_RemoteEstimator,'_client',lambda self:pytest.fail('invalid input reached upload'))
    with pytest.raises(ValueError,match='finite'):Regressor().fit([[0],[1]],[0,np.inf])
    with pytest.raises(ValueError,match='scalar'):table(np.asarray([[{'nested':'invalid'}]],dtype=object))
