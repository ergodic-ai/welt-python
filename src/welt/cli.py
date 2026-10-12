"""Agent-facing commands: JSON stdout, structured errors, explicit tenant selection."""
import argparse
import json
import os
import math
from pathlib import Path
import sys
from uuid import uuid4
import webbrowser

from .client import Client
from .errors import WeltError
from .credentials import secret
from .login import connect, origin, saved_workspaces, use_workspace, logout


def _parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--base-url', default=argparse.SUPPRESS, help='HTTPS API origin (or explicit localhost)')
    common.add_argument('--workspace', default=argparse.SUPPRESS, help='Exact workspace ID or slug; verifies the key tenant')
    common.add_argument('--timeout', type=float, default=argparse.SUPPRESS, help='HTTP or login/wait timeout in seconds')
    parser = argparse.ArgumentParser(prog='welt', parents=[common], description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    def command(name, help):
        return commands.add_parser(name, help=help, parents=[common])

    login = command('login', 'Approve and save a workspace credential in your browser')
    login.add_argument('--no-browser', action='store_true')
    login.add_argument('--no-save', action='store_true')
    command('logout', 'Forget selected local credential; server key remains revocable in console')
    command('whoami', 'Inspect current workspace, role and limits')
    command('pricing', 'Read public pricing; does not purchase')
    command('capabilities', 'Print the machine-readable command tree and authority boundaries')
    console = command('console', 'Open an account-authorized browser control')
    console.add_argument('page', choices=['account', 'workspace', 'members', 'keys', 'billing', 'create'])
    console.add_argument('--no-browser', action='store_true')

    def group(name, help, actions):
        parent = command(name, help)
        subs = parent.add_subparsers(dest='action', required=True)
        return {name: subs.add_parser(name, help=help, parents=[common]) for name, help in actions.items()}

    ws = group('workspaces', 'Manage local workspace connections', dict(list='List saved profiles (no secrets)', current='Inspect credential workspace', use='Select a saved workspace', create='Open workspace creation in browser'))
    ws['use'].add_argument('id')
    ws['create'].add_argument('--no-browser', action='store_true')
    models = group('models', 'Inspect available model capabilities', dict(list='List catalogue', get='Inspect a model from catalogue'))
    models['get'].add_argument('id')
    run = command('run', 'Submit one hidden-model run (tabular, causal or forecasting)')
    run.add_argument('--input', required=True, help='Task request JSON file or - for stdin')
    run.add_argument('--idempotency-key')
    runs = group('model-runs', 'Testing model run lifecycle', dict(delete='Delete terminal input/result; retain metadata and usage'))
    runs['delete'].add_argument('id'); runs['delete'].add_argument('--yes', required=True, action='store_true')
    datasets = group('datasets', 'Private dataset lifecycle', dict(list='List datasets', get='Inspect dataset', upload='Upload table JSON (columns, rows, target, name)', delete='Delete one dataset without dependents'))
    datasets['get'].add_argument('id'); datasets['delete'].add_argument('id'); datasets['delete'].add_argument('--yes', required=True, action='store_true')
    datasets['upload'].add_argument('--input', required=True, help='JSON file or - for stdin')
    uploads = group('uploads', 'Resumable CSV/Parquet file transport', dict(create='Create session before sending bytes', get='Inspect progress', chunk='Send one checksummed chunk', complete='Validate and create dataset', file='Upload or resume a local file'))
    for action in ('create', 'file'):
        uploads[action].add_argument('path'); uploads[action].add_argument('--target'); uploads[action].add_argument('--schema', help='File schema JSON path or -'); uploads[action].add_argument('--format', choices=['csv', 'parquet'])
    uploads['file'].add_argument('--upload-id', help='Resume an exact matching existing upload')
    for action in ('get', 'chunk', 'complete'): uploads[action].add_argument('id')
    uploads['chunk'].add_argument('index', type=int); uploads['chunk'].add_argument('path')
    predictors = group('predictors', 'Inspect reusable predictors', dict(list='List predictors', get='Inspect predictor/version'))
    predictors['get'].add_argument('id')
    jobs = group('jobs', 'Reconnect to durable jobs', dict(list='List jobs', get='Inspect job', cancel='Explicitly cancel job', wait='Wait locally; timeout keeps job running', result='Read ready result without waiting', payload='Read batch payload through API', scores='Read validated causal scores'))
    for action in jobs:
        if action != 'list': jobs[action].add_argument('id')
    jobs['payload'].add_argument('--result-reference')
    jobs['cancel'].add_argument('--yes', required=True, action='store_true')
    for name in ('fit', 'discover'):
        submit = command(name, 'Submit a durable '+name+' job; returns ID and idempotency key')
        submit.add_argument('--dataset', required=True); submit.add_argument('--model', required=True)
        submit.add_argument('--configuration', default='default'); submit.add_argument('--seed', type=int, default=0)
        submit.add_argument('--idempotency-key')
        if name == 'fit': submit.add_argument('--task', required=True, choices=['classification', 'regression'])
        else: submit.add_argument('--constraints', help='Domain constraints JSON file or -'); submit.add_argument('--model-version')
    predict = command('predict', 'Predict a JSON table with a retained predictor')
    predict.add_argument('--predictor', required=True); predict.add_argument('--input', required=True); predict.add_argument('--probabilities', action='store_true')
    batch = command('batch', 'Submit batch prediction on a dataset')
    batch.add_argument('--predictor', required=True); batch.add_argument('--dataset', required=True); batch.add_argument('--probabilities', action='store_true'); batch.add_argument('--idempotency-key')
    usage = command('usage', 'List operation usage; filters are applied locally')
    usage.add_argument('--model'); usage.add_argument('--predictor'); usage.add_argument('--since', help='ISO timestamp inclusive')
    research = group('research', 'Public research catalogue and verified downloads', dict(list='Search one page (follow next_cursor)', get='Inspect research metadata', download='Download verified content to a new file'))
    for option in ('q', 'source', 'availability', 'license', 'cursor'): research['list'].add_argument('--'+option)
    research['list'].add_argument('--limit', type=int, default=20)
    for action in ('get', 'download'): research[action].add_argument('id')
    research['download'].add_argument('path'); research['download'].add_argument('--version'); research['download'].add_argument('--max-bytes', type=int, default=64*1024*1024)
    causal = group('causal-results', 'Retained causal result lifecycle', dict(delete='Delete primary graph/scores; retains metadata'))
    causal['delete'].add_argument('id'); causal['delete'].add_argument('--yes', required=True, action='store_true')
    return parser


def _json_input(path):
    if path == '-': value = json.load(sys.stdin)
    else:
        with Path(path).open() as source: value = json.load(source)
    if not isinstance(value, dict): raise ValueError('JSON input must be an object.')
    return value


def _emit(value, key=None):
    if hasattr(value, 'to_dict'): value = value.to_dict()
    text = json.dumps(value, ensure_ascii=True, allow_nan=False)
    if key: text = text.replace(key, '<redacted>')
    print(text)


def _console(base_url, page, no_browser):
    paths = dict(account='/account', workspace='/workspace', members='/workspace/members', keys='/workspace/keys', billing='/workspace/billing', create='/workspaces/new')
    url = origin(base_url) + paths[page]
    if not no_browser:
        try: webbrowser.open(url)
        except OSError: pass
    return dict(url=url, authority='browser session; sign in and complete the form')


def _tree(parser):
    result = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            result.update({name: _tree(child) for name, child in action.choices.items()})
    if not result:
        return {'arguments': [a.option_strings or [a.dest] for a in parser._actions if a.dest != 'help']}
    return result


def _run(args, client):
    command = args.command
    action = getattr(args, 'action', None)
    if command == 'login':
        connect(client, timeout=args.timeout, open_browser=not args.no_browser, save=not args.no_save, client_name='Welt CLI', workspace=args.workspace, output=sys.stderr)
        return dict(connected=True, workspace=client.workspace())
    if command == 'logout':
        logout(client.base_url, args.workspace or os.getenv('WELT_WORKSPACE'))
        return dict(forgotten=True, server_key_revoked=False)
    if command == 'console': return _console(client.base_url, args.page, args.no_browser)
    if command == 'workspaces':
        if action == 'list': return {'items': saved_workspaces(client.base_url), 'source': 'local saved credentials'}
        if action == 'create': return _console(client.base_url, 'create', args.no_browser)
        if action == 'use':
            # Validate against the backend before changing the active profile.
            from .login import load_saved
            key = load_saved(client.base_url, args.id)
            if key is None: raise ValueError('Run welt login --workspace '+args.id+' first.')
            with Client(base_url=client.base_url, api_key=key, workspace=args.id, timeout=args.timeout) as selected:
                selected.workspace()
            return use_workspace(client.base_url, args.id)
        return client.workspace()
    if command == 'whoami': return client.workspace()
    if command == 'pricing': return client.pricing()
    if command == 'models':
        return {'items': client.models()} if action == 'list' else client.model(args.id)
    if command == 'model-runs':
        return client.delete_model_run(args.id)
    if command == 'run':
        key = args.idempotency_key or uuid4().hex
        args.idempotency_key = key
        print(json.dumps({'submission': {'idempotency_key': key}}), file=sys.stderr)
        job = client.submit_model_run(_json_input(args.input), idempotency_key=key)
        return {'job_id': job.id, 'idempotency_key': key}
    if command == 'datasets':
        if action == 'list': return {'items': client.datasets()}
        if action == 'get': return client.dataset(args.id)
        if action == 'upload': return client.upload(**_json_input(args.input))
        client.delete_dataset(args.id); return {'deleted': args.id}
    if command == 'uploads':
        if action in ('create', 'file'):
            schema = _json_input(args.schema) if args.schema else None
            if action == 'file': return client.upload_file(args.path, target=args.target, schema=schema, format=args.format, upload_id=args.upload_id)
            from .client import _file_spec
            return client.create_upload(**_file_spec(args.path, args.target, schema, args.format))
        if action == 'get': return client.upload_session(args.id)
        if action == 'chunk': return client.put_upload_chunk(args.id, args.index, Path(args.path).read_bytes())
        return client.complete_upload(args.id)
    if command == 'predictors': return {'items': client.predictors()} if action == 'list' else client.predictor(args.id)
    if command == 'jobs':
        if action == 'list': return {'items': client.jobs()}
        if action == 'get': return client.job(args.id).inspect()
        if action == 'cancel': return client.job(args.id).cancel()
        if action == 'wait': return client.job(args.id).result(timeout=args.timeout)
        if action == 'result': return client.job_result(args.id)
        if action == 'scores': return client.causal_scores(args.id)
        return client.batch_payload(args.id, result_reference=args.result_reference)
    if command in ('fit', 'discover', 'batch'):
        key = args.idempotency_key or uuid4().hex
        args.idempotency_key = key
        print(json.dumps({'submission': {'idempotency_key': key}}), file=sys.stderr)
        if command == 'batch': job = client.submit_batch_predict(args.predictor, args.dataset, probabilities=args.probabilities, idempotency_key=key)
        else:
            kwargs = dict(model=args.model, configuration=args.configuration, seed=args.seed, idempotency_key=key)
            if command == 'fit': job = client.submit_fit(args.dataset, task=args.task, **kwargs)
            else: job = client.submit_discover(args.dataset, constraints=_json_input(args.constraints) if args.constraints else None, model_version=args.model_version, **kwargs)
        return {'id': job.id, 'idempotency_key': key}
    if command == 'predict': return client.predict(args.predictor, **_json_input(args.input), probabilities=args.probabilities)
    if command == 'usage':
        items = client.usage()
        if args.since:
            from datetime import datetime
            since = datetime.fromisoformat(args.since.replace('Z', '+00:00'))
            if since.tzinfo is None: raise ValueError('--since requires a timezone.')
            items = [r for r in items if datetime.fromisoformat(r['created_at'].replace('Z', '+00:00')) >= since]
        return {'items': [r for r in items if (not args.model or r.get('model_id') == args.model) and (not args.predictor or r.get('predictor_id') == args.predictor)]}
    if command == 'research':
        if action == 'list': return client.research_datasets(**{k: getattr(args, k) for k in ('q', 'source', 'availability', 'license', 'limit', 'cursor')})
        if action == 'get': return client.research_dataset(args.id)
        return {'path': str(client.download_research_dataset(args.id, args.path, version=args.version, max_bytes=args.max_bytes))}
    if command == 'causal-results': client.delete_causal_result(args.id); return {'deleted': args.id}
    raise ValueError('Unsupported command.')


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    args.base_url = getattr(args, 'base_url', None)
    args.workspace = getattr(args, 'workspace', None)
    args.timeout = getattr(args, 'timeout', 600 if args.command == 'login' or (args.command == 'jobs' and args.action == 'wait') else 60)
    if args.command == 'capabilities':
        _emit({'commands': _tree(parser), 'stdout': 'JSON', 'errors': 'JSON on stderr', 'browser_authority': ['account', 'workspace creation', 'members/invitations', 'keys', 'billing'], 'workspace_selection': 'origin-bound saved credentials; exact ID/slug verified before private operations'})
        return 0
    key = None
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0: raise ValueError('--timeout must be finite and positive.')
        base_url = origin(args.base_url or os.getenv('WELT_BASE_URL', 'https://welt.ergodic.dev'))
        with Client(base_url=base_url, workspace=args.workspace, timeout=args.timeout) as client:
            key = secret(client.api_key)
            result = _run(args, client)
            _emit(result, secret(client.api_key))
    except KeyboardInterrupt:
        print(json.dumps({'error': {'code': 'interrupted', 'message': 'Stopped waiting locally. Accepted server jobs continue; reconnect by ID.'}}), file=sys.stderr)
        return 130
    except (WeltError, ValueError, TypeError, OSError) as error:
        message = str(error)
        if key: message = message.replace(key, '<redacted>')
        value = {'error': dict(code=getattr(error, 'code', None) or 'invalid_input', message=message,
                 status_code=getattr(error, 'status_code', None), request_id=getattr(error, 'request_id', None),
                 job_id=getattr(error, 'job_id', None), retryable=getattr(error, 'retryable', False),
                 retry_after=getattr(error, 'retry_after', None), idempotency_key=getattr(args, 'idempotency_key', None))}
        print(json.dumps(value), file=sys.stderr)
        return 1
    return 0
