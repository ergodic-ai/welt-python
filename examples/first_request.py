"""A real native onboarding request using pinned p10k Iris; held-out evaluation, no benchmark guarantee."""
import argparse
from hashlib import sha256
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from welt import Classifier, Client

ORIGIN = "https://welt.ergodic.dev"
DATASET_ID = "asset-d839644b6de36fc5cafdda18c5daeca3"
VERSION = "d839644b6de36fc5cafdda18c5daeca3284a5ce8bb6169aff2fcb3fbb4a12021"
SIZE_BYTES = 7332
FEATURES = ["petal-length#0", "petal-width#0", "sepal-length#0", "sepal-width#0"]
TARGET = "__target__#0"


def local_digest(path):
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def preparation(frame):
    """Explicit seed9 stratified 120/30 recipe; an explicit teaching split, not a source benchmark."""
    if (frame.shape != (150, 5) or set(frame.columns) != set(FEATURES + [TARGET])
            or frame[FEATURES].isna().any().any() or frame[TARGET].isna().any()
            or frame[TARGET].value_counts().to_dict() != {0: 50, 1: 50, 2: 50}):
        raise RuntimeError("Pinned Iris schema/target does not match the declared onboarding recipe.")
    X_train, X_query, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=30, random_state=9,
        stratify=frame[TARGET])
    return X_train, y_train, X_query, y_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("research")/f"iris-{VERSION}.parquet")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with Client(base_url=ORIGIN) as client:
        client.connect()
        key = client.api_key  # Opaque credential; never printed or saved in this script.
        info = client.research_dataset(DATASET_ID)
        content = info.get("content") or {}
        if (content.get("version") != VERSION or content.get("sha256") != VERSION
                or content.get("size_bytes") != SIZE_BYTES):
            raise RuntimeError("Pinned onboarding research content is not available; inspect the catalogue.")
        if args.output.exists():
            if args.output.is_symlink() or args.output.stat().st_size != SIZE_BYTES or local_digest(args.output) != VERSION:
                raise FileExistsError("Existing output does not match pinned Iris; choose a new --output path.")
        else:
            client.download_research_dataset(DATASET_ID, args.output, version=VERSION)
        print("Research data verified:", DATASET_ID)
        print("License:", info["license"])
        for notice in info["license_notices"]:
            print(notice)
    data = pd.read_parquet(args.output)
    print(data.head())
    print("Rows and columns:", data.shape)
    data.info()
    print(data[TARGET].value_counts())
    X_train, y_train, X_query, y_test = preparation(data)
    model = Classifier(model="tabicl-v2", configuration="default", random_state=9,
                       base_url=ORIGIN, api_key=key)
    print("Preparing a real TabICL v2 classifier on 120 rows and four features…")
    model.fit(X_train, y_train)
    details = model.predict_details(X_query)
    if not isinstance(details, dict) or len(details.get("predictions", [])) != 30:
        raise RuntimeError("Native prediction did not return the declared query-row count.")
    print("Real request complete: 30 predictions received.")
    print("Predictor:", model.predictor_id_, "Model version:", model.model_version_)
    print("Return to the signed-in console and check onboarding for its server-verified milestone.")
    predictions = details["predictions"]
    print(pd.DataFrame({"actual": y_test.to_numpy(), "predicted": predictions}).head())
    print("Test accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=0))
    print("This held-out result is your run, not a guaranteed or publisher benchmark score.")


if __name__ == "__main__":
    main()
