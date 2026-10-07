# Regression and task-specific availability

The SDK uses the same dataset/preparation/predictor lifecycle for classification
and regression. A model family name alone does not imply both tasks are ready.
Read its `task_profiles` entry: regression must be `available`, have a matching
ready worker and report its exact version and limits before preparing a predictor.
The SDK never converts regression into classification or silently selects another
model.

The bounded TabICLv2 regression recipe returns one finite mean point per query row
in target units. Classification classes/probabilities are absent. The underlying
model's native quantile channels are not offered as calibrated intervals or
conformal coverage by this point-prediction interface. Each task has its own
checkpoint, version, qualification and fitted context; a classifier predictor
cannot be reused as a regressor.

Use `Regressor(model="tabicl-v2")` with normal `fit(X, y)`/`predict(X)` when the
service advertises the qualified task. It uses synchronous sklearn conventions.
Existing estimators remain pinned; a new task, seed, configuration or model needs
a new fit. Array inputs use positional schema; named DataFrame columns must match
the fitted names and are safely reordered. Invalid, missing or extra features
reject instead of being silently dropped.

The [regression notebook](https://github.com/ergodic-ai/welt-python/blob/main/examples/notebooks/regression.ipynb)
uses the frozen synthetic planted fixture: seed42, 128 training rows, four numeric
features and 32 query rows. It checks finite point outputs, null class/probability
fields, named-column reordering, pinned version and reopen agreement. It imposes
no scientific accuracy threshold and does not force a worker restart. Default CI
uses a synthetic HTTP fixture; live opt-in requires the separately qualified real
regressor. Accepted synthetic server state remains retained.

Other selected families, task swapping across families, larger envelopes and native
uncertainty outputs require their own exact artifact/runtime qualification. This
example does not complete the entire model catalogue or regression release matrix.
