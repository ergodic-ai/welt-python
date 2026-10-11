"""Workspace selection, local profile isolation and stable agent command contracts."""
from datetime import datetime, timedelta, timezone
import io
import json
import asyncio
import os
from pathlib import Path

import httpx
import pytest
from welt import Client, AsyncClient, AuthenticationError
from welt import cli, login
from test_login import BASE, KEY, initial, connected


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv('WELT_CONFIG_DIR', str(tmp_path/'config'))
    monkeypatch.delenv('WELT_API_KEY', raising=False)
    monkeypatch.delenv('WELT_WORKSPACE', raising=False)
    monkeypatch.delenv('WELT_BASE_URL', raising=False)
    monkeypatch.setattr(login, '_CONNECTED', {})
    monkeypatch.setattr(login, '_PROFILES', {})


def expiry(): return (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
def workspace(id='ws-a', slug='alpha'): return dict(id=id, slug=slug, name=slug, role='member', kind='team')


def test_profiles_preserve_origin_workspace_and_active_selection(monkeypatch):
    login.save_credential(BASE, KEY, expiry(), workspace=workspace())
    login.save_credential(BASE, KEY+'-b', expiry(), workspace=workspace('ws-b', 'beta'))
    assert login.load_saved(BASE, 'alpha') == KEY
    assert login.load_saved(BASE) == KEY+'-b'
    assert login.load_saved('https://foreign.test', 'alpha') is None
    assert login.load_saved(BASE, 'missing') is None
    assert [w['active'] for w in login.saved_workspaces(BASE)] == [False, True]
    assert KEY not in json.dumps(login.saved_workspaces(BASE))
    assert login.use_workspace(BASE, 'alpha')['id'] == 'ws-a'
    assert login.load_saved(BASE) == KEY
    with Client(base_url=BASE, workspace='beta') as c: assert c.api_key == KEY+'-b'
    with Client(base_url=BASE, workspace='missing') as c: assert c.api_key is None
    # Explicit/environment keys remain precedence, but private requests still verify tenant.
    monkeypatch.setenv('WELT_API_KEY', 'explicit-env-key')
    with Client(base_url=BASE, workspace='alpha') as c: assert c.api_key == 'explicit-env-key'
    login.logout(BASE, 'alpha')
    assert login.load_saved(BASE) == KEY+'-b'
    assert login.load_saved(BASE, 'alpha') is None
    login.logout(BASE)
    assert login.saved_workspaces(BASE) == []


def test_selector_device_flow_is_sent_and_mismatch_never_saved():
    requests = []
    def handle(request):
        requests.append(request)
        if request.url.path == '/v1/auth/device':
            assert json.loads(request.content)['workspace'] == 'alpha'
            return httpx.Response(201, json=initial())
        return httpx.Response(200, json=dict(connected(), workspace=workspace('ws-b','beta')))
    with Client(base_url=BASE, transport=httpx.MockTransport(handle)) as c, pytest.raises(AuthenticationError) as e:
        c.connect(workspace='alpha', save=True, open_browser=False)
    assert e.value.code == 'workspace_mismatch'
    assert c.api_key is None and not login.saved_workspaces(BASE)
    assert login._CONNECTED == {}


def test_existing_wrong_workspace_key_starts_exact_selector_flow():
    calls = []
    def handle(request):
        calls.append(request)
        if request.url.path == '/v1/workspace': return httpx.Response(200, json=workspace('ws-b','beta'))
        if request.url.path == '/v1/auth/device': return httpx.Response(201, json=initial())
        return httpx.Response(200, json=dict(connected(), workspace=workspace()))
    with Client(base_url=BASE, api_key='old-key', transport=httpx.MockTransport(handle)) as c:
        c.connect(workspace='alpha', open_browser=False)
        assert c.api_key == KEY
    assert [r.url.path for r in calls] == ['/v1/workspace','/v1/auth/device','/v1/auth/device/token']
    assert json.loads(calls[1].content)['workspace'] == 'alpha'
    assert all('Authorization' not in r.headers for r in calls[1:])


@pytest.mark.parametrize('async_client', [False, True])
def test_selected_key_checked_before_private_payload(async_client):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=workspace('ws-b','beta'))
    async def run():
        async with AsyncClient(base_url=BASE, api_key=KEY, workspace='alpha', transport=httpx.MockTransport(handle)) as c:
            with pytest.raises(AuthenticationError) as e: await c.upload(columns=['x'], rows=[[1]])
            assert e.value.code == 'workspace_mismatch'
    if async_client: asyncio.run(run())
    else:
        with Client(base_url=BASE, api_key=KEY, workspace='alpha', transport=httpx.MockTransport(handle)) as c:
            with pytest.raises(AuthenticationError): c.upload(columns=['x'], rows=[[1]])
    assert [r.url.path for r in calls] == ['/v1/workspace']


@pytest.mark.parametrize('async_client', [False, True])
def test_cross_origin_request_refused_without_sending_key(async_client):
    calls = []
    transport = httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200, json={}))
    async def run():
        async with AsyncClient(base_url=BASE, api_key=KEY, transport=transport) as c:
            with pytest.raises(AuthenticationError): await c.request('GET', 'https://foreign.test/steal')
    if async_client: asyncio.run(run())
    else:
        with Client(base_url=BASE, api_key=KEY, transport=transport) as c:
            with pytest.raises(AuthenticationError): c.request('GET', 'https://foreign.test/steal')
    assert not calls


def test_cli_json_error_redacts_key_and_local_profile_list(monkeypatch, capsys):
    login.save_credential(BASE, KEY, expiry(), workspace=workspace())
    assert cli.main(['workspaces','list','--base-url',BASE]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['source'] == 'local saved credentials'
    assert KEY not in json.dumps(output)
    def handle(request): return httpx.Response(404, json={'error': {'code':'not_found','message':KEY+' failed','request_id':'req-test'}})
    monkeypatch.setattr(cli, 'Client', lambda **kwargs: Client(base_url=BASE, api_key=KEY, transport=httpx.MockTransport(handle)))
    assert cli.main(['datasets','get','unknown']) == 1
    output = capsys.readouterr()
    assert not output.out and KEY not in output.err
    assert json.loads(output.err)['error']['code'] == 'not_found'


def test_cli_capabilities_and_delete_confirmation(capsys):
    assert cli.main(['capabilities']) == 0
    value = json.loads(capsys.readouterr().out)
    assert set(['fit','discover','batch','predict','jobs','uploads']).issubset(value['commands'])
    with pytest.raises(SystemExit) as e: cli.main(['datasets','delete','ds-a'])
    assert e.value.code == 2


def test_cli_fit_preserves_idempotency_and_seed(monkeypatch, capsys):
    calls = []
    def handle(r): calls.append(r); return httpx.Response(202, json={'id':'job-a'})
    monkeypatch.setattr(cli, 'Client', lambda **kwargs: Client(base_url=BASE, api_key=KEY, transport=httpx.MockTransport(handle)))
    assert cli.main(['fit','--dataset','ds-a','--model','tabicl-v2','--task','classification','--seed','19','--idempotency-key','replay-key-123456']) == 0
    assert json.loads(capsys.readouterr().out) == {'id':'job-a','idempotency_key':'replay-key-123456'}
    assert json.loads(calls[0].content)['seed'] == 19
    assert calls[0].headers['Idempotency-Key'] == 'replay-key-123456'


def test_cli_browser_creation_explicit_url_and_no_key_authority(capsys):
    assert cli.main(['workspaces','create','--base-url',BASE,'--no-browser']) == 0
    assert json.loads(capsys.readouterr().out)['url'] == BASE+'/workspaces/new'


@pytest.mark.parametrize('argv', [
    ['--workspace','alpha','datasets','list'],
    ['datasets','--workspace','alpha','list'],
    ['datasets','list','--workspace','alpha'],
])
def test_global_selection_survives_subparsers(argv):
    assert cli._parser().parse_args(argv).workspace == 'alpha'
    assert cli._parser().parse_args(['--timeout','10','login']).timeout == 10
    assert cli._parser().parse_args(['--base-url',BASE,'jobs','list']).base_url == BASE


@pytest.mark.parametrize('mode', ['transport', 'request', 'selection'])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_workspace_guard_rechecks_effective_credential_and_selector(mode, asynchronous):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=workspace() if request.url.path == '/v1/workspace' else {'items': []})
    def mutate(client):
        if mode == 'transport': client.http.headers['Authorization'] = 'Bearer beta-secret'
        if mode == 'selection': client.workspace_selector = 'beta'
        return {'headers': {'Authorization': 'Bearer beta-secret'}} if mode == 'request' else {}
    async def run():
        async with AsyncClient(base_url=BASE, api_key=KEY, workspace='alpha', transport=httpx.MockTransport(handle)) as c:
            await c.datasets()
            with pytest.raises(AuthenticationError): await c.request('POST', '/v1/datasets', json={'columns':['secret'], 'rows':[[42]]}, **mutate(c))
    if asynchronous: asyncio.run(run())
    else:
        with Client(base_url=BASE, api_key=KEY, workspace='alpha', transport=httpx.MockTransport(handle)) as c:
            c.datasets()
            with pytest.raises(AuthenticationError): c.request('POST', '/v1/datasets', json={'columns':['secret'], 'rows':[[42]]}, **mutate(c))
    assert all(r.method == 'GET' for r in calls)


def test_legacy_origin_file_migrates_on_first_selected_login():
    # Legacy file has no tenant metadata: selector construction cannot use it yet.
    login.save_credential(BASE, KEY, expiry())
    def handle(request):
        if request.url.path == '/v1/workspace':
            assert request.headers['Authorization'] == 'Bearer '+KEY
            return httpx.Response(200, json=workspace())
        if request.url.path == '/v1/auth/device': return httpx.Response(201,json=initial())
        return httpx.Response(200,json=dict(connected(),key=KEY+'-b',workspace=workspace('ws-b','beta')))
    with Client(base_url=BASE, workspace='beta', transport=httpx.MockTransport(handle)) as c:
        c.connect(save=True, open_browser=False)
    assert login.load_saved(BASE,'alpha') == KEY
    assert login.load_saved(BASE,'beta') == KEY+'-b'
    assert login.load_saved(BASE) == KEY+'-b'
    assert len(login.saved_workspaces(BASE)) == 2


def test_unavailable_legacy_probe_retains_unknown_credential():
    login.save_credential(BASE, KEY, expiry())
    login.save_credential(BASE, KEY+'-b', expiry(), workspace=workspace('ws-b','beta'))
    state = login._saved(BASE)
    assert state['legacy']['key'] == KEY
    assert login.load_saved(BASE) == KEY+'-b'


def test_concurrent_profiles_serialized_and_logout_never_resurrects(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    original = login._merge_profile
    counts = {'active': 0, 'maximum': 0}
    def merge(*args, **kwargs):
        counts['active'] += 1
        counts['maximum'] = max(counts['maximum'], counts['active'])
        time.sleep(0.02)
        try: return original(*args, **kwargs)
        finally: counts['active'] -= 1
    monkeypatch.setattr(login, '_merge_profile', merge)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(login.save_credential, BASE, KEY+str(i), expiry(), workspace=workspace('ws-'+str(i),'slug-'+str(i))) for i in range(2)]
        for future in pending: future.result()
    assert counts['maximum'] == 1
    assert {w['id'] for w in login.saved_workspaces(BASE)} == {'ws-0','ws-1'}
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(login.logout, BASE, 'ws-0'), pool.submit(login.save_credential, BASE, KEY+'2', expiry(), workspace=workspace('ws-2','slug-2'))]
        for future in pending: future.result()
    assert {w['id'] for w in login.saved_workspaces(BASE)} == {'ws-1','ws-2'}



def test_logout_final_known_profile_keeps_unrelated_legacy_key():
    login.save_credential(BASE, KEY, expiry())
    login.save_credential(BASE, KEY+'-b', expiry(), workspace=workspace('ws-b','beta'))
    login.logout(BASE, 'beta')
    assert login.load_saved(BASE) == KEY
    assert login._saved(BASE).get('legacy') is None
    login.logout(BASE)
    assert login.load_saved(BASE) is None


def test_unsafe_profile_lock_refused(tmp_path):
    folder = Path(os.environ['WELT_CONFIG_DIR']); folder.mkdir(mode=0o700)
    outside = tmp_path/'outside'; outside.write_text('test')
    lock = folder/(login._filename(BASE)+'.lock')
    lock.symlink_to(outside)
    with pytest.raises(AuthenticationError): login.save_credential(BASE, KEY, expiry())
    assert outside.read_text() == 'test'
