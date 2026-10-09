# Predict a number

Classification predicts a label; regression predicts a numeric target. Begin with
[the complete Yacht walkthrough](start.md#regression-predict-a-number): download the
real research table, print its rows/schema, split train/test, fit a `Regressor`,
predict the held-out rows, and calculate MAE, RMSE and R².

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

MAE/RMSE use recorded target units; smaller is better. R² can be negative. Compare
actual and predicted rows to understand misses. These are held-out measurements,
not a publisher benchmark or guarantee. Use a validation split if tuning.

A task-specific available regression profile is required; no task/model fallback
occurs. `predict` returns finite target-unit mean points, not class probabilities,
calibrated intervals or conformal coverage. Predictors keep their original task,
version and configuration. The [notebook](notebooks.md) teaches the full recipe.
