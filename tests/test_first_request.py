"""The declared onboarding recipe uses synthetic rows; never connects to Welt."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def example():
    path=Path(__file__).resolve().parents[1]/'examples/first_request.py'
    spec=importlib.util.spec_from_file_location('first_request',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def table(module):
    frame=pd.DataFrame(np.arange(600,dtype=float).reshape(150,4),columns=module.FEATURES)
    frame[module.TARGET]=np.repeat([0,1,2],50)
    return frame


def test_named_stratified_recipe_is_exact_and_repeatable_without_mutating_source():
    module=example();frame=table(module);before=frame.copy(deep=True)
    X,y,query=module.preparation(frame)
    other_X,other_y,other_query=module.preparation(frame)
    assert X.shape==(120,4) and query.shape==(30,4)
    assert y.value_counts().to_dict()=={0:40,1:40,2:40}
    assert set(X.index).isdisjoint(query.index)
    assert set(X.index)|set(query.index)==set(range(150))
    pd.testing.assert_frame_equal(X,other_X);pd.testing.assert_series_equal(y,other_y)
    pd.testing.assert_frame_equal(query,other_query);pd.testing.assert_frame_equal(frame,before)
    assert module.DATASET_ID=='asset-'+module.VERSION[:32] and module.SIZE_BYTES==7332


@pytest.mark.parametrize('mutation', ['missing_column','missing_value','wrong_rows','wrong_classes'])
def test_recipe_refuses_drift_without_sampling_or_inference(mutation):
    module=example();frame=table(module)
    if mutation=='missing_column':frame=frame.drop(columns=module.FEATURES[0])
    elif mutation=='missing_value':frame.loc[0,module.FEATURES[0]]=np.nan
    elif mutation=='wrong_rows':frame=frame.iloc[:149]
    else:frame[module.TARGET]=0
    with pytest.raises(RuntimeError,match='schema/target'):module.preparation(frame)


def test_cached_file_digest_streaming(tmp_path):
    from hashlib import sha256
    module=example();path=tmp_path/'controlled';body=b'a'*140000;path.write_bytes(body)
    assert module.local_digest(path)==sha256(body).hexdigest()
