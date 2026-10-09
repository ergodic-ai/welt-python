"""API-073: explicit browser/device authentication; no live accounts or secrets."""
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path

import httpx
import pytest

from welt import AuthenticationError, Classifier, Client, PermissionDeniedError
from welt import login

KEY='test-issued-credential-not-a-real-key'
DEVICE='d'*43
CODE='ABCDE-FG234'
BASE='https://welt.test'


@pytest.fixture(autouse=True)
def isolated(monkeypatch,tmp_path):
    monkeypatch.setenv('WELT_CONFIG_DIR',str(tmp_path/'config'))
    monkeypatch.delenv('WELT_API_KEY',raising=False)
    monkeypatch.delenv('WELT_BASE_URL',raising=False)
    monkeypatch.setattr(login,'_CONNECTED',{})


def initial(**updates):
    value=dict(device_code=DEVICE,user_code=CODE,verification_uri=BASE+'/',
        verification_uri_complete=BASE+'/#connect=1&user_code='+CODE,expires_in=600,interval=5)
    return dict(value,**updates)


def connected():
    return dict(status='connected',key=KEY,key_metadata=dict(id='key-test',expires_at=(datetime.now(timezone.utc)+timedelta(days=30)).isoformat()),workspace=dict(id='workspace-test',name='Tests',role='owner'))


def service(responses):
    calls=[]
    def handle(request):
        calls.append(request)
        if request.url.path=='/v1/auth/device':return httpx.Response(201,json=initial())
        if request.url.path=='/v1/auth/device/token':
            assert json.loads(request.content)=={'device_code':DEVICE}
            return responses.pop(0)
        if request.url.path=='/v1/workspace':return httpx.Response(200,json={'id':'workspace-test'})
        raise AssertionError('Unexpected request')
    return httpx.MockTransport(handle),calls


def test_connect_headless_updates_client_and_process_only_cache(monkeypatch):
    transport,calls=service([httpx.Response(200,json=connected())]); output=io.StringIO()
    opened=[];monkeypatch.setattr(login.webbrowser,'open',lambda value:opened.append(value))
    with Client(base_url=BASE,transport=transport) as client:
        assert login.connect(client,open_browser=False,output=output) is client
        assert client.api_key==KEY and client.http.headers['Authorization']=='Bearer '+KEY
        assert len(calls)==2 and all('Authorization' not in request.headers for request in calls)
        with Client(base_url=BASE,transport=transport) as later:assert later.api_key==KEY
        with Client(base_url='https://foreign.test',transport=transport) as foreign:assert foreign.api_key is None
    assert opened==[] and CODE in output.getvalue() and '#connect=1' in output.getvalue()
    assert KEY not in output.getvalue() and DEVICE not in output.getvalue()
    assert not Path(os.environ['WELT_CONFIG_DIR']).exists()


def test_precedence_after_explicit_connection(monkeypatch):
    transport,_=service([httpx.Response(200,json=connected())])
    with Client(base_url=BASE,transport=transport) as client:login.connect(client,open_browser=False,output=io.StringIO())
    monkeypatch.setenv('WELT_API_KEY','environment-key')
    with Client(base_url=BASE,api_key='explicit-key',transport=transport) as client:assert client.api_key=='explicit-key'
    with Client(base_url=BASE,transport=transport) as client:assert client.api_key=='environment-key'
    model=Classifier(base_url=BASE);assert model._client().api_key=='environment-key'


def test_explicit_save_uses_private_atomic_origin_file_then_fresh_process(monkeypatch):
    transport,_=service([httpx.Response(200,json=connected())])
    with Client(base_url=BASE,transport=transport) as client:login.connect(client,save=True,open_browser=False,output=io.StringIO())
    folder=Path(os.environ['WELT_CONFIG_DIR']); files=list(folder.iterdir())
    assert folder.stat().st_mode & 0o777 == 0o700
    assert len(files)==1 and files[0].stat().st_mode & 0o777 == 0o600
    assert json.loads(files[0].read_text())['origin']==BASE
    monkeypatch.setattr(login,'_CONNECTED',{})
    with Client(base_url=BASE,transport=transport) as client:assert client.api_key==KEY
    assert login.load_saved('https://foreign.test') is None


def test_save_existing_process_connection_does_not_issue_another_key():
    transport,calls=service([httpx.Response(200,json=connected())])
    with Client(base_url=BASE,transport=transport) as client:
        login.connect(client,open_browser=False,output=io.StringIO())
        login.connect(client,save=True,open_browser=False,output=io.StringIO())
    assert [r.url.path for r in calls]==['/v1/auth/device','/v1/auth/device/token','/v1/workspace']
    assert login.load_saved(BASE)==KEY


@pytest.mark.parametrize('status,code',[(403,'device_authorization_denied'),(410,'device_authorization_expired'),(409,'device_authorization_consumed'),(404,'invalid_device_code')])
def test_decline_expiry_consumed_are_actionable_without_server_body_reflection(status,code):
    transport,_=service([httpx.Response(status,json={'error':{'message':KEY+DEVICE}})])
    with Client(base_url=BASE,transport=transport) as client,pytest.raises(AuthenticationError) as error:
        login.connect(client,open_browser=False,output=io.StringIO())
    assert error.value.code==code
    assert KEY not in str(error.value) and DEVICE not in str(error.value)
    assert login._CONNECTED=={} and not Path(os.environ['WELT_CONFIG_DIR']).exists()


@pytest.mark.parametrize('update',[
    {'verification_uri_complete':'https://foreign.test/#connect=1&user_code='+CODE},
    {'verification_uri_complete':BASE+'/?secret='+DEVICE+'#connect=1&user_code='+CODE},
    {'verification_uri_complete':BASE+'/#connect=1&user_code='+CODE+'&device_code='+DEVICE},
    {'user_code':'newline\ncode'}, {'interval':0}, {'device_code':'short'},
])
def test_untrusted_verification_links_never_open_or_poll(monkeypatch,update):
    calls=[];opened=[];monkeypatch.setattr(login.webbrowser,'open',lambda link:opened.append(link))
    def handle(request):calls.append(request);return httpx.Response(201,json=initial(**update))
    with Client(base_url=BASE,transport=httpx.MockTransport(handle)) as client,pytest.raises(AuthenticationError,match='unsafe or invalid'):
        login.connect(client,output=io.StringIO())
    assert len(calls)==1 and opened==[]


def test_pending_and_slowdown_waits_respect_local_timeout(monkeypatch):
    clock=[0];sleeps=[]
    monkeypatch.setattr(login.time,'monotonic',lambda:clock[0])
    def sleep(value):sleeps.append(value);clock[0]+=value
    monkeypatch.setattr(login.time,'sleep',sleep)
    transport,_=service([httpx.Response(202,json={'status':'pending','interval':5}),httpx.Response(429,headers={'Retry-After':'30'}),httpx.Response(200,json=connected())])
    with Client(base_url=BASE,transport=transport) as client:login.connect(client,open_browser=False,output=io.StringIO())
    assert sleeps==[5,30]
    transport,_=service([httpx.Response(202,json={'status':'pending','interval':5})])
    with Client(base_url=BASE,transport=transport) as client,pytest.raises(AuthenticationError,match='timed out'):
        client.api_key=None;login.connect(client,timeout=2,open_browser=False,output=io.StringIO())
    assert sleeps[-1]==2


def test_permission_error_on_existing_key_does_not_start_browser_flow():
    calls=[]
    def handle(request):calls.append(request.url.path);return httpx.Response(403,json={'error':{'code':'permission_denied'}})
    with Client(base_url=BASE,api_key='configured-key',transport=httpx.MockTransport(handle)) as client,pytest.raises(PermissionDeniedError):
        client.connect(open_browser=False)
    assert calls==['/v1/workspace']


def test_expired_saved_key_ignored_and_unsafe_file_rejected():
    login.save_credential(BASE,KEY,(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat())
    assert login.load_saved(BASE) is None
    path=next(Path(os.environ['WELT_CONFIG_DIR']).iterdir());path.chmod(0o644)
    with pytest.raises(AuthenticationError,match='private regular files'):login.load_saved(BASE)


def test_browser_failure_keeps_printed_notebook_fallback(monkeypatch):
    def fail(value):raise OSError('no local browser')
    monkeypatch.setattr(login.webbrowser,'open',fail)
    transport,_=service([httpx.Response(200,json=connected())]);out=io.StringIO()
    with Client(base_url=BASE,transport=transport) as client:login.connect(client,output=out)
    assert CODE in out.getvalue() and 'own computer' in out.getvalue()


def test_saved_symlink_file_and_directory_are_rejected(tmp_path,monkeypatch):
    private=tmp_path/'private';private.mkdir(mode=0o700)
    symlink=tmp_path/'linked';symlink.symlink_to(private,target_is_directory=True)
    monkeypatch.setenv('WELT_CONFIG_DIR',str(symlink))
    with pytest.raises(AuthenticationError,match='safely open'):login.load_saved(BASE)
    monkeypatch.setenv('WELT_CONFIG_DIR',str(private))
    outside=tmp_path/'outside';outside.write_text('not a credential')
    (private/login._filename(BASE)).symlink_to(outside)
    with pytest.raises(AuthenticationError,match='safely read'):login.load_saved(BASE)


def test_cancellation_keeps_no_new_credentials(monkeypatch):
    def cancel(value):raise KeyboardInterrupt()
    monkeypatch.setattr(login.time,'sleep',cancel)
    transport,_=service([httpx.Response(202,json={'status':'pending','interval':5})])
    with Client(base_url=BASE,transport=transport) as client,pytest.raises(KeyboardInterrupt):
        login.connect(client,save=True,open_browser=False,output=io.StringIO())
    assert login._CONNECTED=={}
    assert not list(Path(os.environ['WELT_CONFIG_DIR']).glob('*.json'))


def test_explicit_invalid_key_reauth_does_not_send_old_secret_to_device_routes():
    requests=[]
    def handle(request):
        requests.append(request)
        if request.url.path=='/v1/workspace':return httpx.Response(401,json={'error':{'code':'invalid_api_key'}})
        if request.url.path=='/v1/auth/device':return httpx.Response(201,json=initial())
        return httpx.Response(200,json=connected())
    with Client(base_url=BASE,api_key='old-secret',transport=httpx.MockTransport(handle)) as client:
        login.connect(client,open_browser=False,output=io.StringIO())
        assert client.api_key==KEY
    assert requests[0].headers['Authorization']=='Bearer old-secret'
    assert all('Authorization' not in r.headers for r in requests[1:])


def test_cli_success_saves_without_exposing_secrets(monkeypatch,capsys):
    from welt import cli
    transport,_=service([httpx.Response(200,json=connected())])
    monkeypatch.setattr(cli,'Client',lambda **kwargs:Client(base_url=BASE,transport=transport))
    assert cli.main(['login','--no-browser'])==0
    assert login.load_saved(BASE)==KEY
    captured=capsys.readouterr()
    assert CODE in captured.out and KEY not in captured.out+captured.err and DEVICE not in captured.out+captured.err


def test_cli_cancel_has_bounded_plain_error(monkeypatch,capsys):
    from welt import cli
    monkeypatch.setattr(cli,'connect',lambda *a,**k:(_ for _ in ()).throw(KeyboardInterrupt()))
    assert cli.main(['login','--no-save','--no-browser'])==130
    assert 'Stopped waiting locally' in capsys.readouterr().err


@pytest.mark.parametrize('code',['ABCDE-FG289','ZZZZ9-88888'])
def test_backend_alphabet_allows_eight_and_nine(monkeypatch,code):
    monkeypatch.setattr(login.webbrowser,'open',lambda value:True)
    def handle(request):
        if request.url.path=='/v1/auth/device':
            return httpx.Response(201,json=initial(user_code=code,verification_uri_complete=BASE+'/#connect=1&user_code='+code))
        return httpx.Response(200,json=connected())
    with Client(base_url=BASE,transport=httpx.MockTransport(handle)) as client:
        login.connect(client,output=io.StringIO())


@pytest.mark.parametrize('code',['ABCDE-FGI89','ABCDE-FGO89'])
def test_backend_alphabet_rejects_i_and_o(code):
    def handle(request):return httpx.Response(201,json=initial(user_code=code,verification_uri_complete=BASE+'/#connect=1&user_code='+code))
    with Client(base_url=BASE,transport=httpx.MockTransport(handle)) as client,pytest.raises(AuthenticationError,match='unsafe or invalid'):
        login.connect(client,open_browser=False,output=io.StringIO())


def test_existing_key_check_shares_connection_deadline(monkeypatch):
    clock=[0];requests=[]
    monkeypatch.setattr(login.time,'monotonic',lambda:clock[0])
    def handle(request):
        requests.append(request)
        assert request.extensions['timeout']==dict(connect=1.0,read=1.0,write=1.0,pool=1.0)
        clock[0]=2
        return httpx.Response(401,json={'error':{'code':'invalid_api_key'}})
    with Client(base_url=BASE,api_key='configured-key',transport=httpx.MockTransport(handle)) as client,pytest.raises(AuthenticationError,match='timed out'):
        login.connect(client,timeout=1,open_browser=False,output=io.StringIO())
    assert len(requests)==1 and requests[0].url.path=='/v1/workspace'


def test_device_requests_refuse_redirect_even_if_http_client_mutated():
    requests=[]
    def handle(request):
        requests.append(request.url)
        return httpx.Response(307,headers={'Location':'https://foreign.test/steal'})
    with Client(base_url=BASE,transport=httpx.MockTransport(handle)) as client:
        client.http.follow_redirects=True
        with pytest.raises(AuthenticationError,match='Could not start'):login.connect(client,open_browser=False,output=io.StringIO())
    assert len(requests)==1 and requests[0].host=='welt.test'


def test_native_device_flow_never_sends_browser_cookie_or_origin():
    transport,requests=service([httpx.Response(200,json=connected())])
    with Client(base_url=BASE,transport=transport) as client:
        client.http.cookies.set('welt_session','test-browser-secret')
        client.http.headers['Origin']=BASE
        login.connect(client,open_browser=False,output=io.StringIO())
    assert all(not any(header in request.headers for header in ('Cookie','Origin','Authorization')) for request in requests)


def test_mutated_transport_origin_cannot_receive_device_requests():
    requests=[]
    def handle(request):requests.append(request);return httpx.Response(201,json=initial())
    with Client(base_url=BASE,transport=httpx.MockTransport(handle)) as client:
        client.http.base_url='https://foreign.test'
        with pytest.raises(AuthenticationError,match='origin does not match'):
            login.connect(client,open_browser=False,output=io.StringIO())
    assert requests==[]


def test_mutated_transport_origin_cannot_receive_configured_bearer_probe():
    requests=[]
    def handle(request):
        requests.append(request)
        return httpx.Response(200,json={'id':'workspace-test'})
    with Client(base_url=BASE,api_key=KEY,transport=httpx.MockTransport(handle)) as client:
        client.http.base_url='https://foreign.test'
        with pytest.raises(AuthenticationError,match='origin does not match'):
            login.connect(client,open_browser=False,output=io.StringIO())
    assert requests==[]


def test_mutated_http_auth_is_disabled_for_device_requests():
    transport,requests=service([httpx.Response(200,json=connected())])
    with Client(base_url=BASE,transport=transport) as client:
        client.http.auth=httpx.BasicAuth('user','test-only-password')
        login.connect(client,open_browser=False,output=io.StringIO())
    assert all('Authorization' not in request.headers for request in requests)
