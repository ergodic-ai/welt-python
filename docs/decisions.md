# Rank causal interventions

Submit context observations, name a target and the variables you can change, then
receive ranked single-variable actions. This feature requires SDK 0.16.4 or later and a server running the decision
pipeline. Installing this client does not enable server capabilities or establish
production availability.

## Submit and wait

This example uses synthetic observations. `context` fits discovery and estimators;
`rows` asks for personalised rankings without contributing to any fit.

```python file="rank_actions.py"
import numpy as np
import pandas as pd
from welt import Client

rng = np.random.default_rng(8)
z = rng.uniform(-1, 1, 400)
action = rng.integers(0, 3, 400)
context = pd.DataFrame({
    "z": z,
    "action": np.array(["none", "small", "large"])[action],
    "outcome": z + action * (1 + .5 * z) + rng.normal(0, .2, 400),
})

with Client() as client:
    job = client.decisions.submit(
        context,
        target="outcome",
        target_kind="continuous",
        objective="maximize",
        discovery_model="welt-r",
        estimation_model="kumo-medium",
        knowledge={"required_edges": [["z", "outcome"], ["action", "outcome"]]},
        interventions={"action": {"method": "dr", "baseline": "none"}},
        rows=pd.DataFrame({"z": [-.5, .5]}),
        max_fits=100,
    )
    print("Decision job:", job.id)
    result = job.result(timeout=600)
    print(result.population)
    print(result.individual)
    print(result.skipped)
```

Use `client.decisions.create(...)` for submission and waiting together.
`client.decisions.job(job_id).result()` reconnects without resubmitting.
`AsyncClient` exposes the same resource with awaited `submit`, `create`, `result`
and `delete`. A polling timeout leaves the server job running; cancellation is
explicit with `job.cancel()`.

DataFrame, owned dataset ID and named table inputs are accepted for context and
query rows. The existing client workspace/authentication rules apply. Foreign
workspace datasets and jobs remain inaccessible. You can delete terminal payloads
with `client.decisions.delete(job_id)`; metadata and usage remain.

## Choose each variable's method

- Binary and multicategory treatment: `dr`.
- Continuous treatment: `partially_linear` by default, or `continuous_dr`.

Methods are configured inside each variable entry. Without explicit `baseline`
and `values`, boolean treatment uses false→true, categories use the most frequent
class→other classes, and numerical treatment uses median→25th/75th percentiles.
Declare integer-coded categories with `kind="categorical"` or `types`.

For a categorical target, set `target_kind="categorical"` and `target_class` to
the event you want to maximise or minimise. Effects for every class are returned;
labels are never treated as an ordered utility. Binary 0/1 and boolean targets
default to event 1/True. Continuous targets return effects in the target's units.

## Read the result

`population` ranks context-population effects. `individual` preserves query-row
order and gives conditional expected effects for each row's adjustment variables.
These are estimates, not observed individual counterfactuals. `skipped` explains
identification or support refusals. Inspect `result.to_dict()` for native discovery,
the constrained analysis graph, resolved contrasts, model pins and workload.

A contrast uses the same reference for every row; it does not silently substitute
the row's existing treatment. Rankings ignore intervention costs and do not combine
multiple variables into a joint action. Negative improvements remain visible.

Population intervals are pointwise asymptotic intervals and exclude graph-selection,
ranking-selection and smoothing uncertainty. Query effects have no individual
interval. Class-effect vectors sum to zero through a joint score projection that
also propagates covariance. This does not calibrate or clip probability estimates.

Mixed inputs require typed discovery, currently Welt R's admitted native path.
Integer encoding does not make a continuous-only discovery model support categories.
All model versions, capacities and configuration eligibility are checked. Native
nuisance fits use existing fit/predict rates; cross-fitting makes multiple calls.
A conservative plan reserves credits and success settles actual calls. `max_fits`
caps fits; failure/cancellation releases the parent reservation. No deployment or
new model entitlement follows from installing this SDK.
