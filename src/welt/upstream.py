"""Unified upstream data API: explicit previews and verified complete results."""
import asyncio
import copy
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import quote
from uuid import uuid4

import httpx

from .credentials import secret
from datetime import datetime

from .errors import ExecutionError, JobCancelledError, JobTimeoutError, TransportError, WeltError

MAX_BYTES=10*1024*1024


def invalid():
    return WeltError('Upstream result did not match its metadata.',code='invalid_upstream_result')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def ref(value,prefix=None):
    pattern=r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}' if prefix is None else prefix+r'_[0-9a-f]{32}'
    if not isinstance(value,str) or not re.fullmatch(pattern,value):
        raise ValueError('Expected a valid upstream resource identifier.')
    return quote(value,safe='')


def timeout_error(job_id=None):
    return JobTimeoutError('Waiting stopped; retrieval may continue. Reconnect with client.upstream.wait(retrieval_id); do not resubmit.',
                           code='upstream_wait_timeout',job_id=job_id,retryable=True)


def deadline(timeout,poll_interval):
    if (type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0
            or type(poll_interval) not in (int,float) or not math.isfinite(poll_interval) or poll_interval<=0):
        raise ValueError('Timeout and poll interval must be finite and positive.')
    return time.monotonic()+timeout


def finished(status):
    state=status.get('status')
    if state=='succeeded':
        ref(status.get('result_id'),'dres')
        return True
    if state=='failed':
        raise ExecutionError('Upstream retrieval failed; inspect its status before submitting new work.',
                             code=status.get('error_code') or 'retrieval_failed',job_id=status.get('id'))
    if state=='cancelled':
        raise JobCancelledError('Upstream retrieval was cancelled.',code='retrieval_cancelled',job_id=status.get('id'))
    if state not in ('queued','running'):
        raise WeltError('Invalid upstream retrieval status.',code='invalid_response')
    return False


def bounded_selection(description,selection):
    """Use advertised lower ceilings without raising normal client defaults."""
    defaults=dict(max_rows=10000,max_bytes=4194304,max_pages=100,timeout_seconds=30)
    advertised=description.get('limits',{})
    if not isinstance(advertised,dict):
        raise WeltError('Invalid upstream dataset limits.',code='invalid_response')
    for name,value in advertised.items():
        if name not in defaults or type(value) is not int or value<=0:
            raise WeltError('Invalid upstream dataset limits.',code='invalid_response')
        defaults[name]=min(defaults[name],value)
    supplied=selection.get('limits')
    if supplied is not None:
        if not isinstance(supplied,dict):
            raise ValueError('limits must be a mapping.')
        defaults.update(supplied)  # Explicit limits are checked by the service, never silently clamped.
    return dict(selection,limits=defaults)


def payload(dataset_id,schema_revision,*,selections=(),columns=None,as_of=None,limits=None,freshness=None):
    ref(dataset_id)
    if not isinstance(schema_revision,str) or not re.fullmatch('[0-9a-f]{64}',schema_revision):
        raise ValueError('Expected schema_revision from dataset description.')
    result=dict(dataset_id=dataset_id,schema_revision=schema_revision,selections=list(selections))
    for name,value in dict(columns=columns,as_of=as_of,limits=limits,freshness=freshness).items():
        if value is not None:
            result[name]=value
    return result


def replay_key(value):
    if value is None:
        return uuid4().hex
    if not isinstance(value,str) or not re.fullmatch('[A-Za-z0-9_.:-]{1,128}',value):
        raise ValueError('Expected a bounded idempotency key.')
    return value


class UpstreamResult:
    """Complete result with content/snapshot hashes checked against API metadata."""
    def __init__(self,result_id,data,preview):
        ref(result_id,'dres')
        try:
            if preview['id']!=result_id or type(preview['row_count']) is not int:
                raise ValueError()
            snapshot=data['snapshot']
            plan=snapshot['plan']
            columns=plan['columns']
            rows=snapshot['rows']
            if (not isinstance(columns,list) or not 1<=len(columns)<=256
                    or any(not isinstance(c,dict) or not isinstance(c.get('name'),str)
                           or c.get('logical_type') not in ('string','integer','number','boolean','decimal','date','datetime','period')
                           for c in columns)
                    or len({c['name'] for c in columns})!=len(columns)):
                raise ValueError()
            if (plan['contract_version']!='upstream-contract-v1' or snapshot['receipt']['complete'] is not True
                    or not isinstance(rows,list) or len(rows)>100000 or len(rows)!=preview['row_count']
                    or any(not isinstance(row,list) or len(row)!=len(columns) for row in rows)):
                raise ValueError()
            if any(type(cell) not in (str,int,float,bool,type(None)) for row in rows for cell in row):
                raise ValueError()
            content=canonical(dict(contract=plan['contract_version'],columns=columns,rows=rows))
            if len(content)!=snapshot['content_bytes'] or hashlib.sha256(content).hexdigest()!=snapshot['content_sha256']:
                raise ValueError()
            bound={k:v for k,v in snapshot.items() if k!='rows'}
            snapshot_id='snap_'+hashlib.sha256(canonical(bound)).hexdigest()
            if snapshot_id!=preview['snapshot_id'] or {k:v for k,v in data.items() if k!='snapshot'}!=preview['provenance']:
                raise ValueError()
        except (KeyError,TypeError,ValueError,OverflowError):
            raise invalid() from None
        self.id,self.snapshot_id=result_id,snapshot_id
        self._data=copy.deepcopy(data)

    @property
    def columns(self):
        return tuple(c['name'] for c in self._data['snapshot']['plan']['columns'])

    @property
    def rows(self):
        return tuple(tuple(row) for row in self._data['snapshot']['rows'])

    @property
    def provenance(self):
        return copy.deepcopy({k:v for k,v in self._data.items() if k!='snapshot'})

    @property
    def receipt(self):
        return copy.deepcopy(self._data['snapshot']['receipt'])

    def to_dict(self):
        return copy.deepcopy(self._data)

    def to_pandas(self):
        import pandas as pd
        schema=self._data['snapshot']['plan']['columns']
        rows=[[Decimal(cell) if column['logical_type']=='decimal' and cell is not None else cell
               for column,cell in zip(schema,row)] for row in self.rows]
        frame=pd.DataFrame(rows,columns=self.columns,dtype=object)
        frame.attrs['welt_upstream']=dict(result_id=self.id,snapshot_id=self.snapshot_id,
                                         receipt=self.receipt,provenance=self.provenance,columns=copy.deepcopy(schema))
        return frame

    def save(self,path):
        """Publish a complete JSON artifact to a new file; never overwrite a destination."""
        destination=Path(path)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Destination already exists.')
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent,prefix='.welt-upstream-',delete=False) as handle:
                temporary=Path(handle.name)
                handle.write(canonical(self._data))
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary,destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination


def check_headers(response,max_bytes):
    if response.status_code!=200 or response.headers.get('content-encoding','identity').lower()!='identity':
        raise invalid()
    if response.headers.get('content-type','').split(';')[0].strip().lower()!='application/json':
        raise invalid()
    length=response.headers.get('content-length')
    if length is not None and (not length.isascii() or not length.isdigit() or len(length)>12 or int(length)>max_bytes):
        raise invalid()
    return int(length) if length is not None else None


def decode(chunks,expected):
    raw=b''.join(chunks)
    if expected is not None and len(raw)!=expected:
        raise invalid()
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError()
            result[key]=value
        return result
    def constant(_):
        raise ValueError()
    try:
        return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
    except (ValueError,UnicodeError,RecursionError):
        raise invalid() from None


_CONNECTION_SOURCES = ('fred', 'census', 'open_meteo', 'eia', 'comtrade')


def connection_source(source):
    if source not in _CONNECTION_SOURCES:
        raise ValueError('Expected a supported credential source: fred, census, open_meteo, eia or comtrade.')
    return source


def connection_key(source,value):
    connection_source(source)
    value=secret(value)
    pattern={'fred':r'[a-z0-9]{32}','census':r'[a-fA-F0-9]{40}','open_meteo':r'[!-~]{1,128}','eia':r'[!-~]{1,128}','comtrade':r'[a-fA-F0-9]{32}'}[source]
    if not isinstance(value,str) or not re.fullmatch(pattern,value):
        raise ValueError('The credential format is invalid for this source.')
    return value


def connection_metadata(value,source=None,key=None):
    try:
        if type(value) is not dict or set(value)!={'id','source_id','version','created_at','updated_at','revoked_at'}:
            raise ValueError()
        if (type(value['id']) is not str or not re.fullmatch(r'cred_[0-9a-f]{32}',value['id'])
                or value['source_id'] not in _CONNECTION_SOURCES
                or source is not None and value['source_id']!=source
                or type(value['version']) is not int or not 1<=value['version']<=9999999999):
            raise ValueError()
        for name in ('created_at','updated_at','revoked_at'):
            stamp=value[name]
            if name=='revoked_at' and stamp is None:continue
            if (type(stamp) is not str or not re.fullmatch(
                    r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])',stamp)
                    or datetime.fromisoformat(stamp.replace('Z','+00:00')).tzinfo is None):
                raise ValueError()
        # Metadata has only a closed source enum, typed timestamps/version and a
        # fixed-format opaque ID. Short credentials can legitimately occur in
        # these fields; substring matching is not evidence of reflection.
        # Still refuse an opaque ID whose entire payload is the submitted key.
        if key and value['id'][5:]==key:raise ValueError()
        return dict(value)
    except (ValueError,TypeError,KeyError):
        raise WeltError('Source connection metadata is invalid.',code='invalid_source_connection_response') from None


def connection_list(value):
    if not isinstance(value,list) or len(value)>len(_CONNECTION_SOURCES):
        raise WeltError('Source connection metadata is invalid.',code='invalid_source_connection_response')
    rows=[connection_metadata(row) for row in value]
    if len({row['source_id'] for row in rows})!=len(rows):
        raise WeltError('Source connection metadata is invalid.',code='invalid_source_connection_response')
    return rows


def connection_error(error,key):
    # Provider credentials are separate from the platform key redacted by Client.
    # Do not retain arbitrary server messages/details that might reflect this input.
    return type(error)('Source connection could not be updated.',
        code=error.code if key not in str(error.code) else 'source_connection_failed',
        request_id=error.request_id if key not in str(error.request_id) else None,
        retryable=error.retryable,status_code=error.status_code,retry_after=error.retry_after)


class Upstream:
    def __init__(self,client):
        self.client=client

    def connections(self):
        return connection_list(self.client.request('GET','/v1/data/connections',follow_redirects=False,_expected_status=200))

    def connect(self,source_id,*,api_key):
        value=connection_key(source_id,api_key)
        try:
            result=self.client.request('PUT','/v1/data/connections/'+source_id,json={'api_key':value},follow_redirects=False,_expected_status=200)
        except WeltError as error:
            raise connection_error(error,value) from None
        return connection_metadata(result,source_id,value)

    def revoke_connection(self,source_id):
        source_id=connection_source(source_id)
        return connection_metadata(self.client.request('POST','/v1/data/connections/'+source_id+'/revoke',follow_redirects=False,_expected_status=200),source_id)

    def search(self,*,q='',source='',limit=20):
        return self.client.request('GET','/v1/data/datasets',params=dict(q=q,source=source,limit=limit))

    def describe(self,dataset_id):
        return self.client.request('GET','/v1/data/datasets/'+ref(dataset_id))

    def submit(self,dataset_id,*,schema_revision=None,idempotency_key=None,**selection):
        key=replay_key(idempotency_key)
        description=self.describe(dataset_id)
        revision=description['schema_revision'] if schema_revision is None else schema_revision
        return self.client.request('POST','/v1/data/retrievals',
            json=payload(dataset_id,revision,**bounded_selection(description,selection)),headers={'Idempotency-Key':key})

    def status(self,retrieval_id,*,_deadline=None):
        try:
            return self.client.request('GET','/v1/data/retrievals/'+ref(retrieval_id,'dret'),
                                       _deadline=_deadline,_job_id=retrieval_id)
        except JobTimeoutError:
            raise timeout_error(retrieval_id) from None

    def cancel(self,retrieval_id):
        return self.client.request('POST','/v1/data/retrievals/'+ref(retrieval_id,'dret')+'/cancel')

    def preview(self,result_id,*,limit=20):
        return self.client.request('GET','/v1/data/results/'+ref(result_id,'dres'),params={'limit':limit})

    def materialize(self,result_id,*,name,decimal_encoding,target=None,idempotency_key=None):
        """Save an independent modelling dataset; float64 explicitly permits rounding."""
        if decimal_encoding not in ('string','float64'):
            raise ValueError('decimal_encoding must be string or float64.')
        return self.client.request('POST','/v1/data/results/'+ref(result_id,'dres')+'/materialize',
            json=dict(name=name,decimal_encoding=decimal_encoding,target=target),
            headers={'Idempotency-Key':replay_key(idempotency_key)})

    def wait(self,retrieval_id,*,timeout=600,poll_interval=.5):
        until=deadline(timeout,poll_interval)
        while True:
            status=self.status(retrieval_id,_deadline=until)
            if finished(status): return status
            remaining=until-time.monotonic()
            if remaining<=0: raise timeout_error(retrieval_id)
            time.sleep(min(remaining,poll_interval))

    def result(self,result_id,*,max_bytes=MAX_BYTES):
        if type(max_bytes) is not int or not 1<=max_bytes<=MAX_BYTES:
            raise ValueError('max_bytes must be between 1 and 10 MiB.')
        # Preflight uses normal request authentication/workspace checking. The stream
        # stays on the same client and fixed relative API path, without redirects.
        preview=self.preview(result_id,limit=0)
        path='/v1/data/results/'+ref(result_id,'dres')+'/download'
        self.client._authenticate('GET',path)
        chunks,size=[],0
        try:
            with self.client.http.stream('GET',path,headers={'Accept-Encoding':'identity'},follow_redirects=False) as response:
                if response.is_error:
                    from .client import _response,secret
                    raw=b''
                    for chunk in response.iter_raw():
                        raw+=chunk
                        if len(raw)>65536: raise invalid()
                    _response(httpx.Response(response.status_code,headers=response.headers,content=raw),
                              api_key=secret(self.client.api_key))
                expected=check_headers(response,max_bytes)
                for chunk in response.iter_raw():
                    size+=len(chunk)
                    if size>max_bytes: raise invalid()
                    chunks.append(chunk)
        except httpx.RequestError:
            raise TransportError('Upstream download interrupted; no artifact was published.',
                                 code='transport_error',retryable=True) from None
        return UpstreamResult(result_id,decode(chunks,expected),preview)

    def download(self,result_id,path,*,max_bytes=MAX_BYTES):
        if Path(path).exists() or Path(path).is_symlink():
            raise FileExistsError('Destination already exists.')
        return self.result(result_id,max_bytes=max_bytes).save(path)

    def fetch(self,dataset_id,*,timeout=600,poll_interval=.5,max_bytes=MAX_BYTES,**query):
        deadline(timeout,poll_interval)  # Validate before creating remote state.
        if type(max_bytes) is not int or not 1<=max_bytes<=MAX_BYTES:
            raise ValueError('max_bytes must be between 1 and 10 MiB.')
        submitted=self.submit(dataset_id,**query)
        completed=self.wait(submitted['id'],timeout=timeout,poll_interval=poll_interval)
        return self.result(completed['result_id'],max_bytes=max_bytes)


class AsyncUpstream(Upstream):
    async def connections(self):
        return connection_list(await self.client.request('GET','/v1/data/connections',follow_redirects=False,_expected_status=200))

    async def connect(self,source_id,*,api_key):
        value=connection_key(source_id,api_key)
        try:
            result=await self.client.request('PUT','/v1/data/connections/'+source_id,json={'api_key':value},follow_redirects=False,_expected_status=200)
        except WeltError as error:
            raise connection_error(error,value) from None
        return connection_metadata(result,source_id,value)

    async def revoke_connection(self,source_id):
        source_id=connection_source(source_id)
        return connection_metadata(await self.client.request('POST','/v1/data/connections/'+source_id+'/revoke',follow_redirects=False,_expected_status=200),source_id)

    async def search(self,*,q='',source='',limit=20):
        return await self.client.request('GET','/v1/data/datasets',params=dict(q=q,source=source,limit=limit))

    async def describe(self,dataset_id):
        return await self.client.request('GET','/v1/data/datasets/'+ref(dataset_id))

    async def submit(self,dataset_id,*,schema_revision=None,idempotency_key=None,**selection):
        key=replay_key(idempotency_key)
        description=await self.describe(dataset_id)
        revision=description['schema_revision'] if schema_revision is None else schema_revision
        return await self.client.request('POST','/v1/data/retrievals',
            json=payload(dataset_id,revision,**bounded_selection(description,selection)),headers={'Idempotency-Key':key})

    async def status(self,retrieval_id,*,_deadline=None):
        try:
            return await self.client.request('GET','/v1/data/retrievals/'+ref(retrieval_id,'dret'),
                _deadline=_deadline,_job_id=retrieval_id)
        except JobTimeoutError:
            raise timeout_error(retrieval_id) from None

    async def cancel(self,retrieval_id):
        return await self.client.request('POST','/v1/data/retrievals/'+ref(retrieval_id,'dret')+'/cancel')

    async def preview(self,result_id,*,limit=20):
        return await self.client.request('GET','/v1/data/results/'+ref(result_id,'dres'),params={'limit':limit})

    async def materialize(self,result_id,*,name,decimal_encoding,target=None,idempotency_key=None):
        """Save an independent modelling dataset; float64 explicitly permits rounding."""
        if decimal_encoding not in ('string','float64'):
            raise ValueError('decimal_encoding must be string or float64.')
        return await self.client.request('POST','/v1/data/results/'+ref(result_id,'dres')+'/materialize',
            json=dict(name=name,decimal_encoding=decimal_encoding,target=target),
            headers={'Idempotency-Key':replay_key(idempotency_key)})

    async def wait(self,retrieval_id,*,timeout=600,poll_interval=.5):
        until=deadline(timeout,poll_interval)
        while True:
            status=await self.status(retrieval_id,_deadline=until)
            if finished(status): return status
            remaining=until-time.monotonic()
            if remaining<=0: raise timeout_error(retrieval_id)
            await asyncio.sleep(min(remaining,poll_interval))

    async def result(self,result_id,*,max_bytes=MAX_BYTES):
        if type(max_bytes) is not int or not 1<=max_bytes<=MAX_BYTES:
            raise ValueError('max_bytes must be between 1 and 10 MiB.')
        preview=await self.preview(result_id,limit=0)
        path='/v1/data/results/'+ref(result_id,'dres')+'/download'
        self.client._authenticate('GET',path)
        chunks,size=[],0
        try:
            async with self.client.http.stream('GET',path,headers={'Accept-Encoding':'identity'},follow_redirects=False) as response:
                if response.is_error:
                    from .client import _response,secret
                    raw=b''
                    async for chunk in response.aiter_raw():
                        raw+=chunk
                        if len(raw)>65536: raise invalid()
                    _response(httpx.Response(response.status_code,headers=response.headers,content=raw),
                              api_key=secret(self.client.api_key))
                expected=check_headers(response,max_bytes)
                async for chunk in response.aiter_raw():
                    size+=len(chunk)
                    if size>max_bytes: raise invalid()
                    chunks.append(chunk)
        except httpx.RequestError:
            raise TransportError('Upstream download interrupted; no artifact was published.',
                                 code='transport_error',retryable=True) from None
        return UpstreamResult(result_id,decode(chunks,expected),preview)

    async def download(self,result_id,path,*,max_bytes=MAX_BYTES):
        if Path(path).exists() or Path(path).is_symlink():
            raise FileExistsError('Destination already exists.')
        result=await self.result(result_id,max_bytes=max_bytes)
        return await asyncio.to_thread(result.save,path)

    async def fetch(self,dataset_id,*,timeout=600,poll_interval=.5,max_bytes=MAX_BYTES,**query):
        deadline(timeout,poll_interval)
        if type(max_bytes) is not int or not 1<=max_bytes<=MAX_BYTES:
            raise ValueError('max_bytes must be between 1 and 10 MiB.')
        submitted=await self.submit(dataset_id,**query)
        completed=await self.wait(submitted['id'],timeout=timeout,poll_interval=poll_interval)
        return await self.result(completed['result_id'],max_bytes=max_bytes)
