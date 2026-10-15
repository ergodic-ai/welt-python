"""Record the public model catalogue that the documentation Models table renders.

API-122 / DEC-073. The docs build never calls the live service; this explicit,
credential-free read of the public /v1/models route writes a reviewed snapshot that
is committed with the documentation source. Only fields the table renders are kept,
including each task's admitted (rows, columns) shapes (DEC-077).
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import httpx

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'docs/models-snapshot.json'
LIMITS = ('min_rows', 'max_rows', 'max_columns', 'min_features', 'max_features', 'max_classes', 'max_prediction_rows')


def reduce(catalogue, source, taken_at):
    models = []
    for model in catalogue['items']:
        tasks = [dict(task=profile['task'], status=profile['status'],
                      execution_available=bool(profile.get('execution_available')),
                      limits={key: profile['limits'][key] for key in LIMITS if key in (profile.get('limits') or {})},
                      shapes=[dict(rows=shape['rows'], columns=shape['columns']) for shape in profile.get('shapes') or []])
                 for profile in model.get('task_profiles') or []]
        models.append(dict(id=model['id'], name=model['name'], tasks=tasks))
    return dict(source=source, taken_at=taken_at, models=models)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='https://welt.ergodic.dev')
    args = parser.parse_args()
    source = args.base_url.rstrip('/') + '/v1/models'
    response = httpx.get(source, timeout=30, follow_redirects=False)
    response.raise_for_status()
    taken_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    SNAPSHOT.write_text(json.dumps(reduce(response.json(), source, taken_at), indent=2) + '\n')
    print(f'Wrote {SNAPSHOT.relative_to(ROOT)} from {source} at {taken_at}')


if __name__ == '__main__':
    main()
