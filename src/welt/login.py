"""Explicit browser connection and origin-bound local credentials (API-073)."""
from datetime import datetime, timezone
from contextlib import contextmanager
try:
    import fcntl
except ImportError:  # Non-POSIX still supports in-memory/manual credentials.
    fcntl = None
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import time
from urllib.parse import urlsplit, parse_qs
import webbrowser

import httpx

from .credentials import credential, secret
from .errors import AuthenticationError, JobTimeoutError, TransportError, WeltError

_CONNECTED = {}
_PROFILES = {}


def origin(base_url):
    try:
        parts = urlsplit(str(base_url))
        port = parts.port
    except ValueError:
        raise ValueError("Connection requires a valid HTTPS API origin or explicit localhost origin.") from None
    if (parts.scheme not in ('https', 'http') or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment
            or parts.path not in ('', '/') or
            (parts.scheme == 'http' and parts.hostname not in ('localhost', '127.0.0.1', '[::1]', '::1'))):
        raise ValueError('Connection requires an HTTPS API origin, or an explicit local development origin.')
    host = parts.hostname.lower()
    if ':' in host: host = '[' + host + ']'
    suffix = '' if port is None or port == (443 if parts.scheme == 'https' else 80) else ':' + str(port)
    return parts.scheme + '://' + host + suffix


def _expiry(value):
    if not isinstance(value, str): raise ValueError('Invalid credential expiry.')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None: raise ValueError('Credential expiry needs a timezone.')
    return parsed


def _folder():
    return Path(os.environ.get('WELT_CONFIG_DIR', Path.home()/'.config'/'welt'))


def _directory(create=False):
    folder = _folder()
    if create:
        if os.name != 'posix':
            raise AuthenticationError('Secure persisted credentials require POSIX permissions. Use connect(save=False) or WELT_API_KEY.', code='credential_storage_unavailable')
        try: folder.mkdir(parents=True, mode=0o700, exist_ok=True)
        except OSError:
            raise AuthenticationError('Cannot create a private Welt configuration directory. Choose WELT_CONFIG_DIR or use save=False.',code='credential_storage_unavailable') from None
    if not folder.exists(): return None
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    if os.name != 'posix':
        raise AuthenticationError('Secure local credential storage currently requires POSIX permissions. Use connect(save=False) or WELT_API_KEY.', code='credential_storage_unavailable')
    try:
        handle = os.open(folder, flags)
        info = os.fstat(handle)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            os.close(handle)
            raise AuthenticationError('The Welt configuration directory must be owned by you with mode 0700.', code='unsafe_credential_storage')
        return handle
    except OSError:
        raise AuthenticationError('Cannot safely open the Welt configuration directory. Choose a private WELT_CONFIG_DIR or use save=False.', code='unsafe_credential_storage') from None


def _filename(api_origin):
    return hashlib.sha256(api_origin.encode()).hexdigest() + '.json'


def _saved(base_url):
    try: api_origin = origin(base_url)
    except ValueError: return None
    handle = _directory()
    if handle is None: return None
    try:
        try: fd = os.open(_filename(api_origin), os.O_RDONLY | os.O_NOFOLLOW, dir_fd=handle)
        except FileNotFoundError: return None
        except OSError:
            raise AuthenticationError('Cannot safely read the saved Welt credential.', code='unsafe_credential_storage') from None
        with os.fdopen(fd, 'r') as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > 1048576:
                raise AuthenticationError('Saved Welt credentials must be private regular files owned by you, mode 0600.', code='unsafe_credential_storage')
            try:
                value = json.load(source)
                if value['origin'] != api_origin or not isinstance(value['key'], str) or not re.fullmatch(r'[A-Za-z0-9_.-]{16,512}', value['key']):
                    raise ValueError()
                _expiry(value['expires_at'])
                profiles = value.get('profiles', {})
                if not isinstance(profiles, dict): raise ValueError()
                for workspace_id, profile in profiles.items():
                    if (not isinstance(profile, dict) or profile.get('workspace', {}).get('id') != workspace_id
                            or not isinstance(profile.get('key'), str)
                            or not re.fullmatch(r'[A-Za-z0-9_.-]{16,512}', profile['key'])): raise ValueError()
                    _expiry(profile['expires_at'])
                legacy = value.get('legacy')
                if legacy is not None:
                    if not isinstance(legacy, dict) or not isinstance(legacy.get('key'), str) or not re.fullmatch(r'[A-Za-z0-9_.-]{16,512}', legacy['key']): raise ValueError()
                    _expiry(legacy['expires_at'])
                return value
            except (ValueError, KeyError, TypeError):
                raise AuthenticationError('The saved Welt credential is invalid. Choose a new private WELT_CONFIG_DIR and run welt login.', code='invalid_saved_credential') from None
    finally: os.close(handle)


def _matches(profile, workspace):
    metadata = profile.get('workspace') or {}
    return workspace in (metadata.get('id'), metadata.get('slug'))


def validate_workspace(workspace):
    if workspace is not None and (not isinstance(workspace, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', workspace)):
        raise ValueError('workspace must be an exact workspace ID or lowercase slug (1–80 characters).')
    return workspace


def load_saved(base_url, workspace=None):
    validate_workspace(workspace)
    state = _saved(base_url)
    if not state: return None
    profile = state if workspace is None else next((p for p in state.get('profiles', {}).values() if _matches(p, workspace)), None)
    if profile and _expiry(profile['expires_at']) > datetime.now(timezone.utc):
        return credential(profile['key'])
    return None


def selected_credential(base_url, workspace=None):
    validate_workspace(workspace)
    try: api_origin = origin(base_url)
    except ValueError: return None
    current = _CONNECTED.get(api_origin)
    candidates = [p for (o, _), p in _PROFILES.items() if o == api_origin]
    if current: candidates.append(current)
    if workspace is not None: current = next((p for p in candidates if _matches(p, workspace)), None)
    if current and _expiry(current['expires_at']) > datetime.now(timezone.utc): return current['key']
    return load_saved(api_origin, workspace) if os.name == 'posix' else None


def saved_workspaces(base_url):
    """Local credential metadata only; no account-wide server authority is implied."""
    state = _saved(base_url)
    if not state: return []
    return [dict(p['workspace'], expires_at=p['expires_at'],
                 active=secret(state['key']) == p['key'],
                 expired=_expiry(p['expires_at']) <= datetime.now(timezone.utc))
            for p in state.get('profiles', {}).values()]


@contextmanager
def _profile_lock(api_origin):
    """Serialize the complete read/modify/write, across independent CLI processes."""
    handle = _directory(create=True)
    fd = None
    try:
        name = _filename(api_origin) + '.lock'
        try:
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=handle)
        except FileExistsError:
            fd = os.open(name, os.O_RDWR | os.O_NOFOLLOW, dir_fd=handle)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise AuthenticationError('Credential lock must be private and owned by you.', code='unsafe_credential_storage')
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    except OSError:
        raise AuthenticationError('Cannot safely lock workspace credential storage.', code='unsafe_credential_storage') from None
    finally:
        if fd is not None: os.close(fd)
        os.close(handle)


def _merge_profile(api_origin, state, value, expires_at, workspace):
    profiles = state.get('profiles', {})
    profile = dict(key=secret(value), expires_at=expires_at)
    legacy = state.get('legacy')
    if state and not state.get('workspace') and state['key'] != secret(value):
        legacy = dict(key=state['key'], expires_at=state['expires_at'])
    if workspace is not None:
        profile['workspace'] = dict(workspace)
        profiles[workspace['id']] = profile
        if legacy and legacy['key'] == secret(value): legacy = None
    result = dict(profile, origin=api_origin, profiles=profiles)
    if legacy: result['legacy'] = legacy
    _write_saved(api_origin, result)


def use_workspace(base_url, workspace):
    validate_workspace(workspace)
    api_origin = origin(base_url)
    with _profile_lock(api_origin):
        state = _saved(api_origin)
        profile = next((p for p in (state or {}).get('profiles', {}).values() if _matches(p, workspace)), None)
        if not profile or _expiry(profile['expires_at']) <= datetime.now(timezone.utc):
            raise AuthenticationError('No current saved credential for this workspace. Run welt login --workspace ID_OR_SLUG.', code='workspace_login_required')
        _merge_profile(api_origin, state, profile['key'], profile['expires_at'], profile['workspace'])
        _CONNECTED[api_origin] = dict(profile, key=credential(profile['key']))
        return dict(profile['workspace'])


def logout(base_url, workspace=None):
    """Forget a local profile; server revocation remains explicit in the console."""
    validate_workspace(workspace)
    api_origin = origin(base_url)
    with _profile_lock(api_origin):
        state = _saved(api_origin)
        if not state: return
        profiles = state.get('profiles', {})
        selected = next((p for p in profiles.values() if _matches(p, workspace)), None) if workspace else state
        if not selected: return
        for key in [key for key, value in profiles.items() if value['key'] == selected['key']]:
            profiles.pop(key)
            _PROFILES.pop((api_origin, key), None)
        current = _CONNECTED.get(api_origin)
        if current and secret(current['key']) == selected['key']: _CONNECTED.pop(api_origin, None)
        if profiles:
            active = state if state['key'] != selected['key'] else next(iter(profiles.values()))
            result = dict(active, origin=api_origin, profiles=profiles)
            if state.get('legacy') and state['legacy']['key'] != selected['key']:
                result['legacy'] = state['legacy']
            else: result.pop('legacy', None)
            _write_saved(api_origin, result)
        elif state.get('legacy') and state['legacy']['key'] != selected['key']:
            _write_saved(api_origin, dict(state['legacy'], origin=api_origin, profiles={}))
        else:
            handle = _directory()
            try: os.unlink(_filename(api_origin), dir_fd=handle)
            finally: os.close(handle)


def save_credential(base_url, value, expires_at, *, workspace=None):
    api_origin = origin(base_url)
    _expiry(expires_at)
    with _profile_lock(api_origin):
        _merge_profile(api_origin, _saved(api_origin) or {}, value, expires_at, workspace)


def _write_saved(api_origin, state):
    handle = _directory(create=True)
    name = _filename(api_origin)
    temporary = '.' + secrets.token_hex(16) + '.tmp'
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=handle)
        with os.fdopen(fd, 'w') as target:
            json.dump(state, target)
            target.flush(); os.fsync(target.fileno())
        os.replace(temporary, name, src_dir_fd=handle, dst_dir_fd=handle)
        os.fsync(handle)
    except OSError:
        raise AuthenticationError('The connection succeeded but the credential could not be saved. Use the current client or reconnect with save=False.', code='credential_save_failed') from None
    finally:
        try: os.unlink(temporary, dir_fd=handle)
        except FileNotFoundError: pass
        os.close(handle)


def _json(response):
    try:
        value = response.json()
        if not isinstance(value, dict): raise ValueError()
        return value
    except ValueError:
        raise AuthenticationError('Welt returned an invalid connection response.', code='invalid_device_response') from None


def _post(client, path, body, remaining):
    try:
        request = client.http.build_request('POST', path, json=body, timeout=min(30, remaining))
        parts = urlsplit(str(request.url))
        if origin(parts._replace(path='',query='',fragment='').geturl()) != origin(client.base_url):
            raise AuthenticationError('The HTTP transport origin does not match the requested API origin.',code='unsafe_device_origin')
        for header in ('Authorization','Cookie','Origin'):
            request.headers.pop(header, None)
        return client.http.send(request, auth=None, follow_redirects=False)
    except httpx.RequestError:
        raise TransportError('Could not reach Welt during connection. Do not share your device secret; start a new connection when the network is available.', code='device_transport_error') from None


def _link_legacy(client, api_origin, deadline):
    state = _saved(api_origin)
    if not state: return
    legacy = state.get('legacy') or (state if not state.get('workspace') else None)
    if not legacy or _expiry(legacy['expires_at']) <= datetime.now(timezone.utc): return
    request = client.http.build_request('GET', '/v1/workspace', timeout=min(30, max(0.001, deadline-time.monotonic())))
    for header in ('Cookie', 'Origin'): request.headers.pop(header, None)
    request.headers['Authorization'] = 'Bearer ' + legacy['key']
    try: response = client.http.send(request, auth=None, follow_redirects=False)
    except httpx.RequestError: return  # Preserve unknown legacy entry rather than lose it.
    if response.status_code != 200: return
    metadata = _json(response)
    if (not isinstance(metadata.get('id'), str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', metadata['id'])
            or not isinstance(metadata.get('slug'), str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', metadata['slug'])):
        return
    # A simultaneous local logout must not be undone by this read-only probe.
    with _profile_lock(api_origin):
        latest = _saved(api_origin)
        if not latest: return
        unknown = latest.get('legacy') or (latest if not latest.get('workspace') else None)
        if not unknown or unknown['key'] != legacy['key']: return
        profiles = latest.get('profiles', {})
        profile = dict(key=legacy['key'], expires_at=legacy['expires_at'], workspace=metadata)
        profiles[metadata['id']] = profile
        result = dict(latest, profiles=profiles)
        result.pop('legacy', None)
        if latest['key'] == legacy['key']: result['workspace'] = metadata
        _write_saved(api_origin, result)


def connect(client, *, timeout=600, open_browser=True, save=False, client_name='Welt Python SDK', output=None, workspace=None):
    """Explicit connection; no network auth is initiated by constructing a client."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int,float)) or not math.isfinite(timeout) or not 1 <= timeout <= 1800:
        raise ValueError('Connection timeout must be from 1 to 1800 seconds.')
    if type(open_browser) is not bool or type(save) is not bool:
        raise ValueError('open_browser and save must be booleans.')
    requested_workspace = validate_workspace(workspace if workspace is not None else client.workspace_selector)
    workspace = requested_workspace
    api_origin = origin(client.base_url)
    # Check before even the configured-key probe: httpx.base_url is mutable and
    # must never redirect that Bearer credential to a different origin.
    if origin(client.http.base_url) != api_origin:
        raise AuthenticationError('The HTTP transport origin does not match the requested API origin.',
                                  code='unsafe_device_origin')
    output = output or sys.stdout
    deadline = time.monotonic() + timeout
    if save: _link_legacy(client, api_origin, deadline)
    # Existing configured credentials can be reused without creating another key.
    if secret(client.api_key):
        try:
            current_workspace = client.request('GET', '/v1/workspace', _deadline=deadline, timeout=min(30,timeout), _workspace_probe=True)
            if workspace and workspace not in (current_workspace.get('id'), current_workspace.get('slug')):
                raise AuthenticationError('Configured credential belongs to a different workspace.', code='workspace_mismatch')
            client.workspace_selector = workspace
            client._workspace_verified = True
            client._verified_key = client.api_key
            client._verified_workspace = client.workspace_selector
            current = _CONNECTED.get(api_origin)
            if save and current and secret(current['key']) == secret(client.api_key):
                save_credential(api_origin, client.api_key, current['expires_at'], workspace=current.get('workspace'))
            print('Already connected. Configured credentials are reused; no new key was created.', file=output)
            return client
        except JobTimeoutError:
            raise AuthenticationError('Connection timed out while checking the configured credential; no new device connection was started.', code='device_authorization_expired') from None
        except AuthenticationError: pass
    output = output or sys.stdout
    if save:
        handle = _directory(create=True)
        os.close(handle)  # Fail before browser approval if private persistence is unavailable.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AuthenticationError('Connection timed out before browser approval; no new credential was saved.',code='device_authorization_expired')
    initial = _post(client, '/v1/auth/device', {'client_name': client_name, 'scopes':['read','write'], 'expires_in_days':30, **({'workspace': workspace} if workspace else {})}, remaining)
    if initial.status_code != 201:
        raise AuthenticationError('Could not start a connection. Check the service and try again.', code='device_start_failed', status_code=initial.status_code)
    value = _json(initial)
    code = value.get('device_code'); user_code = value.get('user_code')
    interval = value.get('interval'); expires = value.get('expires_in')
    uri = value.get('verification_uri'); complete = value.get('verification_uri_complete')
    try:
        if (not isinstance(code,str) or not re.fullmatch(r'[A-Za-z0-9_-]{43}',code)
                or not isinstance(user_code,str) or not re.fullmatch(r'[A-HJ-NP-Z2-9]{5}-[A-HJ-NP-Z2-9]{5}',user_code)
                or type(interval) is not int or not 1 <= interval <= 60
                or type(expires) is not int or not 1 <= expires <= 600
                or not isinstance(uri,str) or origin(uri) != api_origin
                or not isinstance(complete,str)):
            raise ValueError()
        parts=urlsplit(complete)
        if origin(parts._replace(fragment='').geturl()) != api_origin:
            raise ValueError()
        fragment=parse_qs(parts.fragment,strict_parsing=True)
        if fragment != {'connect':['1'],'user_code':[user_code]}: raise ValueError()
    except (ValueError,TypeError):
        raise AuthenticationError('Welt returned an unsafe or invalid verification link; nothing was opened.', code='invalid_device_response') from None
    deadline = min(deadline, time.monotonic() + expires)
    print('Open this link and approve access to your workspace:', complete, file=output)
    print('Verification code:', user_code, file=output)
    print('If Python is running remotely, open the link on your own computer.', file=output)
    if open_browser:
        try: webbrowser.open(complete)
        except Exception: pass  # The printed link is always a usable fallback.
    while True:
        remaining=deadline-time.monotonic()
        if remaining <= 0:
            raise AuthenticationError('Connection timed out. Start a new connection; no credential was saved.', code='device_authorization_expired')
        response = _post(client,'/v1/auth/device/token',{'device_code':code},remaining)
        if response.status_code == 200:
            result=_json(response); key=result.get('key'); metadata=result.get('key_metadata'); workspace=result.get('workspace')
            try:
                if (result.get('status') != 'connected' or not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{16,512}',key)
                        or not isinstance(metadata,dict) or not isinstance(workspace,dict)
                        or not isinstance(workspace.get('id'),str)
                        or _expiry(metadata['expires_at']) <= datetime.now(timezone.utc)):
                    raise ValueError()
            except (KeyError,ValueError,TypeError):
                raise AuthenticationError('The connection response did not contain a valid expiring workspace credential.', code='invalid_device_response') from None
            if requested_workspace and requested_workspace not in (workspace.get('id'), workspace.get('slug')):
                raise AuthenticationError('Connection returned a different workspace; no credential was installed or saved.', code='workspace_mismatch')
            wrapped=credential(key)
            client.api_key=wrapped
            client.http.headers['Authorization']='Bearer '+key
            _CONNECTED[api_origin]={'key':wrapped,'expires_at':metadata['expires_at'], 'workspace': workspace}
            _PROFILES[(api_origin, workspace['id'])] = _CONNECTED[api_origin]
            client.workspace_selector = requested_workspace
            client._workspace_verified = True
            client._verified_key = client.api_key
            client._verified_workspace = client.workspace_selector
            if save: save_credential(api_origin,wrapped,metadata['expires_at'], workspace=workspace)
            print('Connected to workspace ' + workspace['id'] + '. ' + ('Credential saved privately for this API origin.' if save else 'Credential stays in this Python process only.'),file=output)
            return client
        if response.status_code == 202:
            pending=_json(response)
            if pending.get('status') != 'pending' or type(pending.get('interval')) is not int or not 1 <= pending['interval'] <= 60:
                raise AuthenticationError('Welt returned an invalid pending connection response.',code='invalid_device_response')
            interval=pending['interval']
        elif response.status_code == 429:
            try: interval=float(response.headers['Retry-After'])
            except (KeyError,ValueError): interval=30
            if not math.isfinite(interval) or interval < 1 or interval > 600: interval=30
        else:
            kind={403:('device_authorization_denied','Connection declined in the browser. No credential was saved.'),
                  410:('device_authorization_expired','The verification code expired. Start a new connection.'),
                  409:('device_authorization_consumed','This connection was already completed. Start a new connection if needed.'),
                  404:('invalid_device_code','The connection is no longer available. Start a new connection.')}
            error,message=kind.get(response.status_code,('device_connection_failed','Connection failed. Check the service and start a new connection.'))
            raise AuthenticationError(message,code=error,status_code=response.status_code)
        time.sleep(min(interval,max(0,deadline-time.monotonic())))
