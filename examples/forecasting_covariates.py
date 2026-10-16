"""Forecast simulated store demand with covariates, then score a four-week holdout.

Script version of the forecasting-with-covariates notebook. The data is simulated, so
the scores say nothing about Chronos-2 on your data. The run sends one priced
forecast request to your Welt workspace.
"""
import numpy as np
import pandas as pd
from welt import Forecast

HORIZON = 28  # days held out and forecast
HOLIDAYS = {"2024-12-24": "holiday_eve", "2024-12-25": "holiday", "2024-12-26": "holiday",
            "2024-12-31": "holiday_eve", "2025-01-01": "holiday", "2025-04-18": "holiday",
            "2025-04-21": "holiday", "2025-05-01": "holiday", "2025-08-14": "holiday_eve",
            "2025-08-15": "holiday"}


def simulate(seed=7):
    """Daily sales for five stores: weekly and yearly seasonality, promotions, holidays, weather."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-07-01", "2025-08-24", freq="D")
    season = np.sin(2 * np.pi * (days.dayofyear - 80) / 365.25)  # peaks in late June
    temperature = 13 + 9 * season + rng.normal(0, 2.5, len(days))
    holiday = pd.Series(days.strftime("%Y-%m-%d")).map(HOLIDAYS).fillna("regular").to_numpy()
    holiday_effect = pd.Series(holiday).map({"regular": 1.0, "holiday_eve": 1.35, "holiday": 0.45}).to_numpy()
    stores = {"berlin": (420, 1.25), "hamburg": (310, 1.15), "munich": (380, 1.30),
              "cologne": (240, 1.10), "leipzig": (150, 1.20)}  # base sales, Saturday uplift
    frames = []
    for store, (base, saturday) in stores.items():
        weekly = np.array([0.85, 0.90, 0.95, 1.00, 1.10, saturday, 0.60])[days.dayofweek]
        promo = (rng.random(len(days)) < 0.1).astype(int)  # planned weeks ahead
        trend = 1 + 0.0004 * np.arange(len(days))
        mean = (base * trend * weekly * (1 + 0.08 * season) * (1 + 0.4 * promo) * holiday_effect
                * (1 + 0.01 * (temperature - 13)))
        frames.append(pd.DataFrame({"date": days, "store": store, "sales": rng.poisson(mean).astype(float),
                                    "promo": promo, "holiday": holiday, "temperature": temperature.round(1)}))
    return pd.concat(frames, ignore_index=True)


def seasonal_naive(history, test, m=7):
    """Repeat each store's last week: the forecast to beat."""
    parts = []
    for store, past in history.groupby("store", sort=False):
        last_week = past.sort_values("date")["sales"].to_numpy()[-m:]
        dates = test.loc[test["store"] == store, "date"].sort_values().to_numpy()
        parts.append(pd.DataFrame({"store": store, "date": dates, "baseline": np.resize(last_week, len(dates))}))
    return pd.concat(parts, ignore_index=True)


def mase(actual, forecast, past, m=7):
    """Mean absolute error divided by the in-sample seasonal-naive error (m = 7 days)."""
    return np.mean(np.abs(actual - forecast)) / np.mean(np.abs(past[m:] - past[:-m]))


def main():
    data = simulate()
    cutoff = data["date"].max() - pd.Timedelta(days=HORIZON)
    history, test = data[data["date"] <= cutoff], data[data["date"] > cutoff]
    future = test[["date", "store", "promo", "holiday"]]  # known in advance; no sales, no temperature

    model = Forecast(model="chronos-2").fit(
        history, time="date", id="store", target=["sales"],
        past_covariates=["temperature"], future_covariates=["promo", "holiday"], categorical=["holiday"])
    result = model.forecast(HORIZON, future=future, quantile_levels=[0.1, 0.5, 0.9])
    frame = result.to_frame().rename(columns={"id": "store"})
    frame["date"] = frame["timestamp"].dt.tz_localize(None)

    scored = (test.merge(seasonal_naive(history, test), on=["store", "date"])
              .merge(frame[["store", "date", "median", "q_0.1", "q_0.9"]], on=["store", "date"]))
    rows = []
    for store, part in scored.groupby("store", sort=False):
        past = history.loc[history["store"] == store, "sales"].to_numpy()
        inside = (part["sales"] >= part["q_0.1"]) & (part["sales"] <= part["q_0.9"])
        rows.append({"store": store, "seasonal naive": mase(part["sales"], part["baseline"], past),
                     "Chronos-2": mase(part["sales"], part["median"], past), "80% coverage": inside.mean()})
    per_store = pd.DataFrame(rows).set_index("store")
    print(per_store.round(3))
    print("Mean MASE, seasonal naive:", round(per_store["seasonal naive"].mean(), 3))
    print("Mean MASE, Chronos-2:", round(per_store["Chronos-2"].mean(), 3))
    print("80% interval coverage:", round(per_store["80% coverage"].mean(), 3), "(intervals are not calibrated)")
    print("Sorted quantile steps:", result.rearranged_steps, "· Job:", result.job_id)


if __name__ == "__main__":
    main()
