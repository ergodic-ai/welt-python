"""SDK 0.7 first prediction: explicit task, frozen synthetic input, durable reuse.

Run after configuring WELT_API_KEY locally. This makes a real hosted fit/predict
request, retains its synthetic dataset/predictor and never deletes server state.
"""
import argparse
import getpass
import os

import numpy as np
import pandas as pd
from welt import Classifier, Regressor


def example_tables(task="classification"):
    """Preserve the documented seed9 classification / seed42 regression recipes."""
    rng = np.random.default_rng(9 if task == "classification" else 42)
    values = rng.normal(size=(160, 4))
    train = pd.DataFrame(values[:128], columns=["a", "b", "c", "d"])
    query = pd.DataFrame(values[128:], columns=train.columns)
    if task == "classification":
        train["label"] = np.where(values[:128, 0] + 0.5 * values[:128, 1] > 0,
                                  "positive", "negative")
    else:
        weights = rng.normal(size=4)
        train["target"] = values[:128] @ weights + rng.normal(scale=0.05, size=128)
    return train, query


def run(task="classification"):
    train, query = example_tables(task)
    estimator = Classifier if task == "classification" else Regressor
    model = estimator(model="tabicl-v2", random_state=9 if task == "classification" else 42)
    model.fit(train, target="label" if task == "classification" else "target")
    predictions = model.predict(query)
    assert predictions.shape == (32,)
    reopened = estimator.from_predictor(model.predictor_id_)
    repeated = reopened.predict(query)
    if task == "classification":
        np.testing.assert_array_equal(predictions, repeated)
    else:
        np.testing.assert_allclose(predictions, repeated, atol=1e-5, rtol=1e-5)
    print(f"{task}: numpy.ndarray, shape={predictions.shape}")
    print(f"Reusable predictor: {model.predictor_id_}")
    return model, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression"), default="classification")
    args = parser.parse_args()
    if not os.environ.get("WELT_API_KEY"):
        os.environ["WELT_API_KEY"] = getpass.getpass("Welt API key (hidden): ")
    run(args.task)


if __name__ == "__main__":
    main()
