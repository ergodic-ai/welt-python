"""Cross-validate models on a local CSV with a target column; provide your own table."""
import argparse
from pathlib import Path

import pandas as pd
from welt import Client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('table', help='CSV with feature columns and a target column')
    parser.add_argument('--target', required=True)
    parser.add_argument('--task', choices=('classification', 'regression'), required=True)
    parser.add_argument('--models', nargs='*', default=None, help='Model IDs; omit to let Welt choose')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--report', default='evaluation-report.html')
    args = parser.parse_args()
    data = pd.read_csv(args.table)
    with Client() as client:
        evaluation = client.evaluate(data, args.target, task=args.task, models=args.models, folds=args.folds)
        print('Evaluation:', evaluation.id, evaluation.status)
        print(evaluation.summary())
        for model in evaluation.skipped:
            print('Skipped:', model['model_id'], model['reason'])
        Path(args.report).write_text(evaluation.report_html(), encoding="utf-8")
        print('Report saved to', args.report)


if __name__ == '__main__':
    main()
