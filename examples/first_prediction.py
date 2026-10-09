"""Learn classification or regression with real Welt research data.

Run with WELT_API_KEY configured, or enter a key at the hidden prompt.
Requires welt-client[parquet]. Each run retains its hosted training dataset/predictor.
"""
import argparse
import getpass
import os


def run(task="classification"):
    if task == "classification":
        from pathlib import Path
        from tempfile import mkdtemp
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        from welt import Client, Classifier

        # 1. Get a real table from Welt's research inventory.
        dataset_id = "asset-d839644b6de36fc5cafdda18c5daeca3"
        version = "d839644b6de36fc5cafdda18c5daeca3284a5ce8bb6169aff2fcb3fbb4a12021"
        path = Path(mkdtemp(prefix="welt-iris-")) / "iris.parquet"
        with Client() as client:
            info = client.research_dataset(dataset_id)
            print("Dataset:", info["name"], "— License:", info["license"])
            for notice in info["license_notices"]:
                print(notice)
            client.download_research_dataset(dataset_id, path, version=version)
        data = pd.read_parquet(path)
        target = "__target__#0"  # The target recorded in this research snapshot.

        # 2. Look at the rows before modelling.
        print(data.head())
        print("Rows and columns:", data.shape)
        data.info()
        print(data[target].value_counts())

        # 3. Set aside test rows before fitting; never train on their answers.
        train, test = train_test_split(
            data, test_size=0.2, random_state=9, stratify=data[target])
        query = test.drop(columns=target)
        y_test = test[target]
        print("Training rows:", len(train), "— Test rows:", len(test))

        # 4. Create a model and train it on the training table.
        model = Classifier(model="tabicl-v2", random_state=9)
        model.fit(train, target=target)

        # 5. Predict the held-out rows, without passing their target column.
        predictions = model.predict(query)
        print(pd.DataFrame({"actual": y_test.to_numpy(), "predicted": predictions}).head())

        # 6. Compare predictions with the answers we kept aside.
        print("Test accuracy:", accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions, zero_division=0))
        print("Keep this predictor ID:", model.predictor_id_)
    else:
        from pathlib import Path
        from tempfile import mkdtemp
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
        from welt import Client, Regressor

        # 1. Get a real table from Welt's research inventory.
        dataset_id = "asset-bc7dddd884c5e798c1b2abfc559a36ab"
        version = "bc7dddd884c5e798c1b2abfc559a36ab79a6d09c4e2ea3e9bbd95c06539c7342"
        path = Path(mkdtemp(prefix="welt-yacht-")) / "yacht.parquet"
        with Client() as client:
            info = client.research_dataset(dataset_id)
            print("Dataset:", info["name"], "— License:", info["license"])
            for notice in info["license_notices"]:
                print(notice)
            client.download_research_dataset(dataset_id, path, version=version)
        data = pd.read_parquet(path)
        target = "__target__#0"  # The target recorded in this research snapshot.

        # 2. Look at the rows before modelling.
        print(data.head())
        print("Rows and columns:", data.shape)
        data.info()
        print(data[target].describe())

        # 3. Set aside test rows before fitting; never train on their answers.
        train, test = train_test_split(
            data, test_size=0.2, random_state=9)
        query = test.drop(columns=target)
        y_test = test[target]
        print("Training rows:", len(train), "— Test rows:", len(test))

        # 4. Create a model and train it on the training table.
        model = Regressor(model="tabicl-v2", random_state=9)
        model.fit(train, target=target)

        # 5. Predict the held-out rows, without passing their target column.
        predictions = model.predict(query)
        print(pd.DataFrame({"actual": y_test.to_numpy(), "predicted": predictions}).head())

        # 6. Compare predictions with the answers we kept aside.
        print("Test MAE:", mean_absolute_error(y_test, predictions))
        print("Test RMSE:", root_mean_squared_error(y_test, predictions))
        print("Test R²:", r2_score(y_test, predictions))
        print("Keep this predictor ID:", model.predictor_id_)
    return model, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression"), default="classification")
    args = parser.parse_args()
    if not os.environ.get("WELT_API_KEY"):
        os.environ["WELT_API_KEY"] = getpass.getpass("Welt API key: ")
    run(args.task)


if __name__ == "__main__":
    main()
