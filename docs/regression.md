# Predict a number

Classification predicts a label; regression predicts a numeric target. Here you fit a
`Regressor` on real Yacht data and measure its errors on held-out rows.

!!! need "You'll need"
    - The Yacht train and test tables from
      [the complete Yacht walkthrough](start.md#regression-predict-a-number). It
      downloads the real research table, prints its rows and schema, and splits train
      and test rows.

## Fit and measure

Fit a `Regressor`, predict the held-out rows, then calculate MAE, RMSE and R²:

```python
# Continue after preparing the Yacht train/test tables in Start.
from welt import Regressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

model = Regressor(model="tabicl-v2", random_state=9)
model.fit(train, target=target)
predictions = model.predict(query)
print("Test MAE:", mean_absolute_error(y_test, predictions))
print("Test RMSE:", root_mean_squared_error(y_test, predictions))
print("Test R²:", r2_score(y_test, predictions))
```

- **MAE** and **RMSE** use the recorded target's units; smaller is better.
- **R²** can be negative.

Compare actual and predicted rows to understand the misses.

!!! caution "Your measurements, not a benchmark"
    These are held-out measurements, not a publisher benchmark or a guarantee. If you
    tune, use a separate validation split.

## What a regressor returns

`predict` returns finite mean points in the target's units. It does not return class
probabilities, calibrated intervals or conformal coverage.

A task-specific available regression profile is required; there is no task or model
fallback. Predictors keep their original task, version and configuration.

## Next steps

<div class="cards" markdown="1">

- [Notebook examples](notebooks.md): the full regression recipe in a notebook.
- [Reuse a predictor](reuse.md): reopen a fitted predictor without another fit.
- [Use your own table](own-data.md): run the same workflow on your data.

</div>
