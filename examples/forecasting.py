"""Forecast an explicitly timestamped local CSV; provide your own real observations."""
import argparse

import pandas as pd
from welt import Forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history', help='CSV containing 16–512 regular observations')
    parser.add_argument('--time', default='timestamp')
    parser.add_argument('--target', default='value')
    parser.add_argument('--horizon', type=int, default=24)
    args = parser.parse_args()
    history = pd.read_csv(args.history)
    # Preserve explicit offsets; the SDK rejects naive or irregular timestamps.
    history[args.time] = pd.to_datetime(history[args.time])
    forecast = Forecast().fit(history, time=args.time, target=args.target)
    result = forecast.predict_details(args.horizon)
    print(result.to_frame())
    print('Job:', result.job_id, 'Model:', result.model_version, 'Configuration:', result.configuration)


if __name__ == '__main__':
    main()
