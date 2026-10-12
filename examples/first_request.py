"""Run a first Iris classification request with the current research table."""
import argparse
from hashlib import sha256
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from welt import Classifier, Client

ORIGIN = "https://welt.ergodic.dev"
DATASET_ID = "asset-d839644b6de36fc5cafdda18c5daeca3"
FEATURES = ["petal-length", "petal-width", "sepal-length", "sepal-width"]
TARGET = "target"


def local_digest(path):
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def preparation(frame):
    """Seed9 stratified120/30 teaching split; keep all four measured features."""
    if (frame.shape != (150, 5) or set(frame.columns) != set(FEATURES + [TARGET])
            or frame.isna().any().any()
            or frame[TARGET].value_counts().to_dict() != {0: 50, 1: 50, 2: 50}):
        raise RuntimeError("Iris schema/target does not match the declared teaching recipe.")
    X_train, X_query, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=30, random_state=9,
        stratify=frame[TARGET])
    return X_train, y_train, X_query, y_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("research") / "iris.parquet")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with Client(base_url=ORIGIN) as client:
        client.connect()
        key = client.api_key
        info = client.research_dataset(DATASET_ID)
        content = info.get("content") or {}
        if content.get("media_type") != "application/vnd.apache.parquet":
            raise RuntimeError("Iris currently has no direct Parquet download.")
        if args.output.exists():
            if (args.output.is_symlink() or args.output.stat().st_size != content["size_bytes"]
                    or local_digest(args.output) != content["sha256"]):
                raise FileExistsError("Existing output differs from current Iris; choose a new --output path.")
        else:
            client.download_research_dataset(DATASET_ID, args.output)
        print("Research data:", info["name"])
    data = pd.read_parquet(args.output)
    print(data.head())
    print("Rows and columns:", data.shape)
    data.info()
    print(data[TARGET].value_counts())
    X_train, y_train, X_query, y_test = preparation(data)
    model = Classifier(model="tabicl-v2", random_state=9, base_url=ORIGIN, api_key=key)
    print("Preparing a classifier on 120 rows and four features…")
    model.fit(X_train, y_train)
    details = model.predict_details(X_query)
    if not isinstance(details, dict) or len(details.get("predictions", [])) != 30:
        raise RuntimeError("Prediction did not return the expected query-row count.")
    print("Request complete: 30 predictions received.")
    print("Predictor:", model.predictor_id_)
    predictions = details["predictions"]
    print(pd.DataFrame({"actual": y_test.to_numpy(), "predicted": predictions}).head())
    print("Test accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
