"""Equivalent synchronous/asynchronous resources and durable job handles."""
import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from pathlib import Path
import math
import os
import time
from urllib.parse import quote
from uuid import uuid4
import httpx
from .credentials import credential, secret
from .errors import (AuthenticationError, CapacityError, ConflictError, ExecutionError,
    InvalidInputError, JobCancelledError, JobTimeoutError, ModelUnavailableError,
    NotFoundError, PermissionDeniedError, PredictionPendingError, RateLimitError,
    ResultExpiredError, ResultDeletedError, TransportError, WeltError, safe_id)


def _retry_after(response):
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            delay = (when - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0, delay) if math.isfinite(delay) else None


def _error_fields(error, api_key=None):
    fields = dict(error)
    if api_key:
        fields = {k: (None if isinstance(v, str) and api_key in v else v)
                  for k, v in fields.items()}
    fields["code"] = safe_id(fields.get("code")) or "unknown"
    fields["retryable"] = fields.get("retryable") is True
    return fields


def _response(response, *, api_key=None):
    if response.is_error:
        try:
            error = response.json()["error"]
            if not isinstance(error, dict):
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            if response.status_code in (401, 403):
                error = {"code": "authentication_failed" if response.status_code == 401 else "permission_denied"}
            else:
                raise WeltError(f"HTTP {response.status_code} from Welt API.",
                                status_code=response.status_code) from None
        code = _error_fields(error, api_key)["code"]
        classes = {401: AuthenticationError, 403: PermissionDeniedError,
                   404: NotFoundError, 409: ConflictError, 410: ResultExpiredError,
                   422: InvalidInputError, 429: RateLimitError}
        cls = classes.get(response.status_code, WeltError)
        if code in ("invalid_input", "schema_mismatch", "workload_limit", "payload_limit",
                    "unsupported_capability"):
            cls = InvalidInputError
        elif code in ("model_unavailable", "version_unavailable", "worker_unavailable"):
            cls = ModelUnavailableError
        elif code == "capacity_limit":
            cls = CapacityError
        elif code == "execution_failed":
            cls = ExecutionError
        elif code == "result_deleted":
            cls = ResultDeletedError
        elif code == "prediction_timeout":
            cls = PredictionPendingError
        message = error.get("message", "Welt request failed.")
        if response.status_code == 401:
            cls = AuthenticationError
            state = "expired" if isinstance(code, str) and "expired" in code else "invalid or revoked"
            message = (f"The API key is {state}. Create a current workspace key in Welt → API keys "
                       "and set WELT_API_KEY outside your code or notebook.")
        elif response.status_code == 403:
            cls = PermissionDeniedError
            message = ("This key lacks permission for this operation. Use a key for the correct workspace "
                       "with the required read/write permissions; fitting requires write permission.")
        if api_key:
            message = str(message).replace(api_key, "<redacted>")
        error = _error_fields(error, api_key)
        raise cls(message, code=code,
                  request_id=error.get("request_id"), retryable=error.get("retryable", False),
                  job_id=error.get("job_id"), status_code=response.status_code,
                  retry_after=_retry_after(response))
    if response.status_code == 204:
        return None
    try:
        return response.json()
    except ValueError:
        raise WeltError("Welt API returned an invalid response.", code="invalid_response",
                        status_code=response.status_code) from None


def _id(value):
    return quote(str(value), safe="")


def _fit(dataset_id, model, task, configuration, seed):
    return dict(dataset_id=dataset_id, model_id=model, task=task,
                configuration=configuration, seed=seed)


def _discovery(dataset_id, model, configuration, seed, constraints, model_version):
    if type(seed) is not int or not 0 <= seed <= 2**32-1:
        raise ValueError("seed must be an unsigned32-bit integer.")
    return dict(dataset_id=dataset_id, model_id=model, model_version=model_version,
                configuration=configuration, seed=seed,
                constraints=constraints if constraints is not None else
                dict(required_edges=[], forbidden_edges=[], temporal_order=[]))


def _file_spec(path, target, schema, format):
    path = Path(path)
    format = format or path.suffix.lower().lstrip('.')
    if format not in ('csv', 'parquet'): raise ValueError('File format must be csv or parquet.')
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            size += len(block)
            digest.update(block)
    if not size: raise ValueError('Upload file must not be empty.')
    return dict(format=format, size_bytes=size, sha256=digest.hexdigest(), target=target, schema=schema)


def _check_upload(upload, spec):
    if not isinstance(upload, dict) or any(upload.get(k) != v for k, v in spec.items()):
        raise ValueError('Upload metadata does not match this file, schema and target; nothing was uploaded.')
    chunk = upload.get('chunk_bytes')
    received = upload.get('received_indices')
    if (type(chunk) is not int or not 1 <= chunk <= 64 * 1024 * 1024
            or not isinstance(received, list) or any(type(i) is not int or i < 0 for i in received)):
        raise WeltError('Invalid upload response.', code='invalid_response')
    if upload.get('status') not in ('uploading', 'complete'):
        raise ValueError('Upload is not writable. Start a new upload.')
    return chunk, set(received)


class _HTTP:
    def _configure(self, base_url, api_key, max_retries, max_retry_delay, workspace=None):
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be an integer from 0 to 5.")
        if not math.isfinite(max_retry_delay) or not 0 <= max_retry_delay <= 60:
            raise ValueError("max_retry_delay must be from 0 to 60 seconds.")
        self.base_url = base_url if base_url is not None else os.getenv("WELT_BASE_URL", "https://welt.ergodic.dev")
        from .login import selected_credential, validate_workspace
        self.workspace_selector = validate_workspace(workspace if workspace is not None else os.getenv('WELT_WORKSPACE'))
        self._workspace_verified = False
        self._verified_key = None
        self._verified_workspace = None
        self.api_key = credential(secret(api_key) or os.getenv("WELT_API_KEY") or selected_credential(self.base_url, self.workspace_selector))
        self.max_retries, self.max_retry_delay = max_retries, max_retry_delay
        value = secret(self.api_key)
        return {"Authorization": f"Bearer {value}"} if value else {}

    def _authenticate(self, method, path, headers=None, auth=None):
        configured = httpx.URL(self.base_url)
        actual = self.http.base_url
        destination = httpx.URL(path)
        identity = lambda url: (url.scheme, url.host, url.port)
        if identity(actual) != identity(configured) or (destination.is_absolute_url and identity(destination) != identity(configured)):
            raise AuthenticationError('The HTTP transport origin does not match the requested API origin.', code='unsafe_request_origin')
        expected = 'Bearer ' + secret(self.api_key) if secret(self.api_key) else None
        effective = httpx.Headers(headers or {}).get('Authorization', self.http.headers.get('Authorization'))
        if effective != expected or (expected and (auth is not None or self.http.auth is not None)):
            raise AuthenticationError('Authorization overrides cannot change the configured workspace credential.', code='workspace_credential_override')
        # Only intentionally public catalogue reads bypass credential preflight.
        route = httpx.URL(path).path
        public = method.upper() == "GET" and (
            route in ("/v1/models", "/v1/billing/pricing") or route == "/v1/research-datasets" or
            (route.startswith("/v1/research-datasets/") and route.count("/") == 3))
        value = secret(self.api_key)
        if not public and (not value or not value.strip()):
            raise AuthenticationError(
                "Run welt login or Client().connect() before this operation, or set WELT_API_KEY from Welt → API keys. "
                "Keep the key outside your code or notebook; fitting requires write permission.",
                code="missing_api_key")

    def _budget(self, kwargs, deadline, job_id):
        if deadline is None:
            return kwargs
        remaining = _wait(deadline, job_id, float("inf"))
        configured = httpx.Timeout(kwargs.get("timeout", self.http.timeout))
        values = {phase: remaining if value is None else min(value, remaining)
                  for phase, value in configured.as_dict().items()}
        return dict(kwargs, timeout=httpx.Timeout(**values))

    def _delay(self, method, attempt, response=None):
        # Reads retry automatically. Mutations require caller-controlled replay.
        if method.upper() != "GET" or attempt >= self.max_retries:
            return None
        if response is not None and response.status_code not in (429, 502, 503, 504):
            return None
        delay = _retry_after(response) if response is not None else None
        delay = min(0.25 * 2**attempt, self.max_retry_delay) if delay is None else delay
        return delay if delay <= self.max_retry_delay else None


class Client(_HTTP):
    def __init__(self, base_url=None, api_key=None, *, timeout=60, transport=None,
                 max_retries=2, max_retry_delay=5, workspace=None):
        headers = self._configure(base_url, api_key, max_retries, max_retry_delay, workspace)
        self.http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout,
                                 transport=transport, follow_redirects=False)

    def connect(self, *, timeout=600, open_browser=True, save=False, workspace=None):
        """Approve browser sign-in; retain access in-process or explicitly save it.

        Returns this connected Client. Constructors never open a browser. The link
        and short code also work from remote notebooks/headless machines.
        """
        from .login import connect
        return connect(self, timeout=timeout, open_browser=open_browser, save=save, workspace=workspace)

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def request(self, method, path, *, _deadline=None, _job_id=None, _workspace_probe=False, **kwargs):
        self._authenticate(method, path, kwargs.get("headers"), kwargs.get("auth"))
        if not _workspace_probe and self.workspace_selector and (not self._workspace_verified or self._verified_key != self.api_key or self._verified_workspace != self.workspace_selector) and not (method.upper() == "GET" and (httpx.URL(path).path in ("/v1/models", "/v1/billing/pricing", "/v1/research-datasets") or (httpx.URL(path).path.startswith("/v1/research-datasets/") and httpx.URL(path).path.count("/") == 3))):
            current = self.request("GET", "/v1/workspace", _workspace_probe=True, _deadline=_deadline, _job_id=_job_id)
            if self.workspace_selector not in (current.get('id'), current.get('slug')):
                raise AuthenticationError('Credential does not belong to the selected workspace.', code='workspace_mismatch')
            self._workspace_verified = True
            self._verified_key = self.api_key
            self._verified_workspace = self.workspace_selector
        value = secret(self.api_key)
        if value and _job_id is not None and value in str(_job_id):
            _job_id = None
        for attempt in range(self.max_retries + 1):
            budgeted = self._budget(kwargs, _deadline, _job_id)
            try:
                response = self.http.request(method, path, **budgeted)
            except httpx.RequestError:
                if _deadline is not None:
                    _wait(_deadline, _job_id, float("inf"))
                delay = self._delay(method, attempt)
                if delay is None:
                    raise TransportError("Could not reach Welt API; reconnect to accepted jobs before repeating a mutation.",
                                         code="transport_error", retryable=True, job_id=_job_id) from None
            else:
                if _deadline is not None:
                    _wait(_deadline, _job_id, float("inf"))
                delay = self._delay(method, attempt, response)
                if delay is None:
                    result = _response(response, api_key=secret(self.api_key))
                    if _deadline is not None:
                        _wait(_deadline, _job_id, float("inf"))
                    return result
            time.sleep(delay if _deadline is None else _wait(_deadline, _job_id, delay))

    def workspace(self):
        """Inspect the credential's tenant, role, limits and resource capacity."""
        return self.request("GET", "/v1/workspace")

    def pricing(self):
        """Read the public rate card and available packs/plans; does not purchase."""
        return self.request("GET", "/v1/billing/pricing")

    def create_upload(self, *, format, size_bytes, sha256, target=None, schema=None):
        return self.request("POST", "/v1/uploads", json=dict(
            format=format, size_bytes=size_bytes, sha256=sha256, target=target, schema=schema))

    def upload_session(self, upload_id):
        return self.request("GET", f"/v1/uploads/{_id(upload_id)}")

    def put_upload_chunk(self, upload_id, index, content):
        if type(index) is not int or index < 0 or not isinstance(content, bytes):
            raise ValueError('A nonnegative chunk index and bytes content are required.')
        return self.request("PUT", f"/v1/uploads/{_id(upload_id)}/chunks/{index}", content=content,
            headers={"Content-Type": "application/octet-stream", "X-Chunk-SHA256": hashlib.sha256(content).hexdigest()})

    def complete_upload(self, upload_id):
        return self.request("POST", f"/v1/uploads/{_id(upload_id)}/complete")

    def upload_file(self, path, *, target=None, schema=None, format=None, upload_id=None):
        """Upload bounded chunks; resume only against matching immutable file metadata.

        Keep upload_id from create_upload() for recovery after a transport failure.
        No mutation is silently retried. Completion is server-side checksum/schema validated.
        """
        spec = _file_spec(path, target, schema, format)
        upload = self.upload_session(upload_id) if upload_id else self.create_upload(**spec)
        chunk, received = _check_upload(upload, spec)
        if upload['status'] == 'complete': return self.dataset(upload['dataset_id'])
        with Path(path).open('rb') as source:
            for index in range((spec['size_bytes'] + chunk - 1) // chunk):
                body = source.read(chunk)
                if index not in received: self.put_upload_chunk(upload['id'], index, body)
        return self.complete_upload(upload['id'])

    def submit_batch_predict(self, predictor_id, dataset_id, *, probabilities=False, idempotency_key=None):
        result = self.request("POST", "/v1/batch-predictions",
            json=dict(predictor_id=predictor_id, dataset_id=dataset_id, probabilities=probabilities),
            headers={"Idempotency-Key": idempotency_key or uuid4().hex})
        return Job(self, result['id'])

    def job_result(self, job_id):
        """Retrieve a completed result without polling; not-ready/expired errors stay explicit."""
        return self.request("GET", f"/v1/jobs/{_id(job_id)}/result")

    def batch_payload(self, job_id, *, result_reference=None):
        """Retrieve JSON payload through the authenticated API, never a third-party URL."""
        if result_reference is None:
            result_reference = self.job_result(job_id)['result_reference']
        return self.request("GET", f"/v1/jobs/{_id(job_id)}/result/payload", params=dict(result_reference=result_reference))

    def models(self):
        return self.request("GET", "/v1/models")["items"]

    def research_datasets(self, *, q=None, source=None, availability=None, license=None,
                          limit=20, cursor=None):
        """Search one bounded research-catalogue page; does not upload content."""
        from .research import search_params
        return self.request("GET", "/v1/research-datasets", params=search_params(
            q, source, availability, license, limit, cursor))

    def research_dataset(self, dataset_id):
        from .research import research_id
        return self.request("GET", f"/v1/research-datasets/{research_id(dataset_id)}")

    def download_research_dataset(self, dataset_id, path, *, version=None,
                                  max_bytes=64 * 1024 * 1024, metadata=False):
        """Download pinned bytes to a new file; verify size/SHA before publication.

        Returns the path. With metadata=True, also writes the catalogue detail to
        ``<path>.metadata.json`` and returns a ResearchDownload.
        """
        from pathlib import Path
        from .research import (ResearchDownload, download_metadata, metadata_destination,
                               publish_metadata, refuse_existing, research_id)
        if type(metadata) is not bool:
            raise ValueError("metadata must be True or False.")
        dataset_id = research_id(dataset_id)
        if metadata:  # Refuse either destination before any request or partial write.
            refuse_existing(path, metadata_destination(path))
        detail = self.research_dataset(dataset_id)
        pinned, digest, size = download_metadata(detail, dataset_id, version, max_bytes)
        self._stream_research(dataset_id, path, pinned, digest, size)
        if not metadata:
            return Path(path)
        sidecar = publish_metadata(metadata_destination(path), detail)
        return ResearchDownload(Path(path), sidecar, pinned, digest, detail)

    def load_research_dataset(self, dataset_id, *, version=None, metadata=False,
                              max_bytes=64 * 1024 * 1024):
        """Download verified content privately and return it as a pandas DataFrame.

        Returns a DataFrame, or (DataFrame, ResearchMetadata) with metadata=True.
        Needs the ``welt-client[research]`` extra; no file is left behind.
        """
        import tempfile
        from pathlib import Path
        from .research import download_metadata, read_table, require_table_extra, research_id
        if type(metadata) is not bool:
            raise ValueError("metadata must be True or False.")
        pandas = require_table_extra()
        dataset_id = research_id(dataset_id)
        detail = self.research_dataset(dataset_id)
        pinned, digest, size = download_metadata(detail, dataset_id, version, max_bytes)
        with tempfile.TemporaryDirectory(prefix="welt-research-") as folder:
            path = Path(folder) / "content"
            self._stream_research(dataset_id, path, pinned, digest, size)
            frame, info = read_table(pandas, path, detail, dataset_id, pinned, digest, max_bytes)
        return (frame, info) if metadata else frame

    def _stream_research(self, dataset_id, path, pinned, digest, size):
        from .research import destination
        try:
            with destination(path, digest, size) as writer:
                with self.http.stream("GET", f"/v1/research-datasets/{dataset_id}/content",
                         params={"version": pinned}, headers={"Accept-Encoding": "identity"},
                         follow_redirects=False) as response:
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        from .research import invalid_download
                        raise invalid_download()
                    if response.is_error:
                        error_body = bytearray()
                        for chunk in response.iter_raw(chunk_size=65536):
                            if len(error_body) + len(chunk) > 65536:
                                from .research import invalid_download
                                raise invalid_download()
                            error_body.extend(chunk)
                        _response(httpx.Response(response.status_code, headers=response.headers,
                                                 content=bytes(error_body)))
                    writer.headers(response)
                    for chunk in response.iter_raw(chunk_size=65536):
                        writer.write(chunk)
        except httpx.RequestError:
            raise TransportError("Research download interrupted; no destination was published.",
                                 code="transport_error", retryable=True) from None

    def datasets(self):
        return self.request("GET", "/v1/datasets")["items"]

    def predictors(self):
        return self.request("GET", "/v1/predictors")["items"]

    def jobs(self):
        return self.request("GET", "/v1/jobs")["items"]

    def usage(self):
        return self.request("GET", "/v1/usage")["items"]

    def dataset(self, dataset_id):
        return self.request("GET", f"/v1/datasets/{_id(dataset_id)}")

    def delete_dataset(self, dataset_id):
        """Delete this owned dataset only if it has no retained/active dependents."""
        return self.request("DELETE", f"/v1/datasets/{_id(dataset_id)}")

    def delete_causal_result(self, result_id):
        """Explicitly delete an owned primary graph/scores; retain job/usage metadata."""
        return self.request("DELETE", f"/v1/causal-results/{_id(result_id)}")

    def upload(self, *, columns, rows, name="SDK dataset", target=None):
        return self.request("POST", "/v1/datasets", json=dict(
            columns=columns, rows=rows, name=name, target=target))

    def submit_fit(self, dataset_id, *, model, task, configuration="default", seed=0,
                   idempotency_key=None):
        result = self.request("POST", "/v1/fits", json=_fit(
            dataset_id, model, task, configuration, seed),
            headers={"Idempotency-Key": idempotency_key or uuid4().hex})
        return Job(self, result["id"])

    def submit_discover(self, dataset_id, *, model="arrow", configuration="default", seed=0,
                        constraints=None, model_version=None, idempotency_key=None):
        payload = _discovery(dataset_id, model, configuration, seed, constraints, model_version)
        result = self.request("POST", "/v1/discoveries", json=payload,
                              headers={"Idempotency-Key": idempotency_key or uuid4().hex})
        return Job(self, result["id"])

    def discover(self, dataset_id, *, model="arrow", configuration="default", seed=0,
                 constraints=None, model_version=None, idempotency_key=None, timeout=600):
        return self.submit_discover(dataset_id, model=model, configuration=configuration,
            seed=seed, constraints=constraints, model_version=model_version,
            idempotency_key=idempotency_key).result(timeout=timeout)

    def causal_result(self, job_id, *, _deadline=None):
        from .causal import CausalResult
        from .errors import InvalidCausalResultError
        result = CausalResult(self.request("GET", f"/v1/jobs/{_id(job_id)}/result", _deadline=_deadline, _job_id=job_id))
        if result.job_id != str(job_id):
            raise InvalidCausalResultError("Causal result belongs to a different discovery job.", code="invalid_causal_result")
        return result

    def causal_scores(self, job_id):
        from .causal import validated_scores
        result = self.causal_result(job_id)
        return validated_scores(result, self.request("GET", f"/v1/jobs/{_id(job_id)}/result/scores"))

    def job(self, job_id):
        return Job(self, job_id)

    def predictor(self, predictor_id, *, _deadline=None, _job_id=None):
        return self.request("GET", f"/v1/predictors/{_id(predictor_id)}", _deadline=_deadline, _job_id=_job_id)

    def predict(self, predictor_id, *, columns, rows, probabilities=False):
        return self.request("POST", f"/v1/predictors/{_id(predictor_id)}/predict",
                            json=dict(columns=columns, rows=rows, probabilities=probabilities))


def _failed(job, api_key=None):
    if job["status"] in ("failed", "cancelled"):
        error = job.get("error") or {"code": "job_cancelled", "message": "Job was cancelled."}
        message = str(error.get("message", "Job failed."))
        if api_key:
            message = message.replace(api_key, "<redacted>")
        error = _error_fields(error, api_key)
        job_id = job["id"] if not api_key or api_key not in str(job["id"]) else None
        cls = JobCancelledError if job["status"] == "cancelled" else ExecutionError
        raise cls(message, code=error["code"], job_id=job_id,
                  request_id=error.get("request_id"), retryable=error.get("retryable", False))


def _deadline(timeout, poll_interval):
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout must be finite and nonnegative.")
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("poll_interval must be finite and positive.")
    return time.monotonic() + timeout


def _timeout(job_id):
    from .errors import safe_id
    display = safe_id(job_id)
    reconnect = f"Client().job('{display}').result()" if display else "Client().job(job_id).result()"
    return JobTimeoutError(
        f"Waiting stopped; the server job continues. Reconnect with {reconnect} "
        "using the same endpoint and current workspace credentials. Do not resubmit the fit.",
        code="job_wait_timeout", job_id=job_id, retryable=True)


def _wait(deadline, job_id, interval):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _timeout(job_id)
    return min(interval, remaining)


class Job:
    def __init__(self, client, job_id):
        value = secret(client.api_key)
        if value and value in str(job_id):
            raise WeltError("Cannot use a job identifier containing the configured credential. "
                            "Inspect jobs before repeating a submission.", code="invalid_response")
        self.client, self.id = client, job_id

    def inspect(self):
        return self.client.request("GET", f"/v1/jobs/{_id(self.id)}")

    def cancel(self):
        return self.client.request("POST", f"/v1/jobs/{_id(self.id)}/cancel")

    def result(self, *, timeout=600, poll_interval=0.25):
        """Wait locally, including reads/retries; timeout leaves the server job running.

        HTTP phases use the remaining budget. Blocking synchronous transports
        cannot be forcibly interrupted; elapsed time is checked after each read.
        """
        deadline = _deadline(timeout, poll_interval)
        while True:
            job = self.client.request("GET", f"/v1/jobs/{_id(self.id)}", _deadline=deadline, _job_id=self.id)
            if job["status"] in ("succeeded", "partially_completed"):
                if job["status"] == "partially_completed" and job["operation"] != "batch_predict":
                    raise WeltError("Unexpected partial result for this job type.", code="invalid_response")
                if job["operation"] == "discover":
                    return self.client.causal_result(self.id, _deadline=deadline)
                return (self.client.request("GET", f"/v1/jobs/{_id(self.id)}/result", _deadline=deadline, _job_id=self.id)
                        if job["operation"] in ("predict", "batch_predict") else self.client.predictor(job["predictor_id"], _deadline=deadline, _job_id=self.id))
            _failed(job, secret(self.client.api_key))
            time.sleep(_wait(deadline, self.id, poll_interval))
            poll_interval = min(poll_interval * 1.4, 5)


class AsyncClient(_HTTP):
    def __init__(self, base_url=None, api_key=None, *, timeout=60, transport=None,
                 max_retries=2, max_retry_delay=5, workspace=None):
        headers = self._configure(base_url, api_key, max_retries, max_retry_delay, workspace)
        self.http = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout,
                                      transport=transport, follow_redirects=False)

    async def aclose(self):
        await self.http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def request(self, method, path, *, _deadline=None, _job_id=None, _workspace_probe=False, **kwargs):
        self._authenticate(method, path, kwargs.get("headers"), kwargs.get("auth"))
        if not _workspace_probe and self.workspace_selector and (not self._workspace_verified or self._verified_key != self.api_key or self._verified_workspace != self.workspace_selector) and not (method.upper() == "GET" and (httpx.URL(path).path in ("/v1/models", "/v1/billing/pricing", "/v1/research-datasets") or (httpx.URL(path).path.startswith("/v1/research-datasets/") and httpx.URL(path).path.count("/") == 3))):
            current = await self.request("GET", "/v1/workspace", _workspace_probe=True, _deadline=_deadline, _job_id=_job_id)
            if self.workspace_selector not in (current.get('id'), current.get('slug')):
                raise AuthenticationError('Credential does not belong to the selected workspace.', code='workspace_mismatch')
            self._workspace_verified = True
            self._verified_key = self.api_key
            self._verified_workspace = self.workspace_selector
        value = secret(self.api_key)
        if value and _job_id is not None and value in str(_job_id):
            _job_id = None
        for attempt in range(self.max_retries + 1):
            budgeted = self._budget(kwargs, _deadline, _job_id)
            try:
                if _deadline is None:
                    response = await self.http.request(method, path, **budgeted)
                else:
                    remaining = _wait(_deadline, _job_id, float("inf"))
                    try:
                        async with asyncio.timeout(remaining):
                            response = await self.http.request(method, path, **budgeted)
                    except TimeoutError:
                        raise _timeout(_job_id) from None
            except httpx.RequestError:
                if _deadline is not None:
                    _wait(_deadline, _job_id, float("inf"))
                delay = self._delay(method, attempt)
                if delay is None:
                    raise TransportError("Could not reach Welt API; reconnect to accepted jobs before repeating a mutation.",
                                         code="transport_error", retryable=True, job_id=_job_id) from None
            else:
                if _deadline is not None:
                    _wait(_deadline, _job_id, float("inf"))
                delay = self._delay(method, attempt, response)
                if delay is None:
                    result = _response(response, api_key=secret(self.api_key))
                    if _deadline is not None:
                        _wait(_deadline, _job_id, float("inf"))
                    return result
            await asyncio.sleep(delay if _deadline is None else _wait(_deadline, _job_id, delay))

    async def workspace(self):
        """Inspect the credential's tenant, role, limits and resource capacity."""
        return await self.request("GET", "/v1/workspace")

    async def pricing(self):
        """Read the public rate card and available packs/plans; does not purchase."""
        return await self.request("GET", "/v1/billing/pricing")

    async def create_upload(self, *, format, size_bytes, sha256, target=None, schema=None):
        return await self.request("POST", "/v1/uploads", json=dict(
            format=format, size_bytes=size_bytes, sha256=sha256, target=target, schema=schema))

    async def upload_session(self, upload_id):
        return await self.request("GET", f"/v1/uploads/{_id(upload_id)}")

    async def put_upload_chunk(self, upload_id, index, content):
        if type(index) is not int or index < 0 or not isinstance(content, bytes):
            raise ValueError('A nonnegative chunk index and bytes content are required.')
        return await self.request("PUT", f"/v1/uploads/{_id(upload_id)}/chunks/{index}", content=content,
            headers={"Content-Type": "application/octet-stream", "X-Chunk-SHA256": hashlib.sha256(content).hexdigest()})

    async def complete_upload(self, upload_id):
        return await self.request("POST", f"/v1/uploads/{_id(upload_id)}/complete")

    async def upload_file(self, path, *, target=None, schema=None, format=None, upload_id=None):
        """Upload bounded chunks; resume only against matching immutable file metadata.

        Keep upload_id from create_upload() for recovery after a transport failure.
        No mutation is silently retried. Completion is server-side checksum/schema validated.
        """
        spec = _file_spec(path, target, schema, format)
        upload = await self.upload_session(upload_id) if upload_id else await self.create_upload(**spec)
        chunk, received = _check_upload(upload, spec)
        if upload['status'] == 'complete': return await self.dataset(upload['dataset_id'])
        with Path(path).open('rb') as source:
            for index in range((spec['size_bytes'] + chunk - 1) // chunk):
                body = source.read(chunk)
                if index not in received: await self.put_upload_chunk(upload['id'], index, body)
        return await self.complete_upload(upload['id'])

    async def submit_batch_predict(self, predictor_id, dataset_id, *, probabilities=False, idempotency_key=None):
        result = await self.request("POST", "/v1/batch-predictions",
            json=dict(predictor_id=predictor_id, dataset_id=dataset_id, probabilities=probabilities),
            headers={"Idempotency-Key": idempotency_key or uuid4().hex})
        return AsyncJob(self, result['id'])

    async def job_result(self, job_id):
        """Retrieve a completed result without polling; not-ready/expired errors stay explicit."""
        return await self.request("GET", f"/v1/jobs/{_id(job_id)}/result")

    async def batch_payload(self, job_id, *, result_reference=None):
        """Retrieve JSON payload through the authenticated API, never a third-party URL."""
        if result_reference is None:
            result_reference = (await self.job_result(job_id))['result_reference']
        return await self.request("GET", f"/v1/jobs/{_id(job_id)}/result/payload", params=dict(result_reference=result_reference))

    async def models(self):
        return (await self.request("GET", "/v1/models"))["items"]

    async def research_datasets(self, *, q=None, source=None, availability=None, license=None,
                                limit=20, cursor=None):
        from .research import search_params
        return await self.request("GET", "/v1/research-datasets", params=search_params(
            q, source, availability, license, limit, cursor))

    async def research_dataset(self, dataset_id):
        from .research import research_id
        return await self.request("GET", f"/v1/research-datasets/{research_id(dataset_id)}")

    async def download_research_dataset(self, dataset_id, path, *, version=None,
                                        max_bytes=64 * 1024 * 1024, metadata=False):
        """Async Client.download_research_dataset with the same contract."""
        from pathlib import Path
        from .research import (ResearchDownload, download_metadata, metadata_destination,
                               publish_metadata, refuse_existing, research_id)
        if type(metadata) is not bool:
            raise ValueError("metadata must be True or False.")
        dataset_id = research_id(dataset_id)
        if metadata:  # Refuse either destination before any request or partial write.
            refuse_existing(path, metadata_destination(path))
        detail = await self.research_dataset(dataset_id)
        pinned, digest, size = download_metadata(detail, dataset_id, version, max_bytes)
        await self._stream_research(dataset_id, path, pinned, digest, size)
        if not metadata:
            return Path(path)
        sidecar = publish_metadata(metadata_destination(path), detail)
        return ResearchDownload(Path(path), sidecar, pinned, digest, detail)

    async def load_research_dataset(self, dataset_id, *, version=None, metadata=False,
                                    max_bytes=64 * 1024 * 1024):
        """Async Client.load_research_dataset with the same contract."""
        import tempfile
        from pathlib import Path
        from .research import download_metadata, read_table, require_table_extra, research_id
        if type(metadata) is not bool:
            raise ValueError("metadata must be True or False.")
        pandas = require_table_extra()
        dataset_id = research_id(dataset_id)
        detail = await self.research_dataset(dataset_id)
        pinned, digest, size = download_metadata(detail, dataset_id, version, max_bytes)
        with tempfile.TemporaryDirectory(prefix="welt-research-") as folder:
            path = Path(folder) / "content"
            await self._stream_research(dataset_id, path, pinned, digest, size)
            frame, info = read_table(pandas, path, detail, dataset_id, pinned, digest, max_bytes)
        return (frame, info) if metadata else frame

    async def _stream_research(self, dataset_id, path, pinned, digest, size):
        from .research import destination
        try:
            with destination(path, digest, size) as writer:
                async with self.http.stream("GET", f"/v1/research-datasets/{dataset_id}/content",
                         params={"version": pinned}, headers={"Accept-Encoding": "identity"},
                         follow_redirects=False) as response:
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        from .research import invalid_download
                        raise invalid_download()
                    if response.is_error:
                        error_body = bytearray()
                        async for chunk in response.aiter_raw(chunk_size=65536):
                            if len(error_body) + len(chunk) > 65536:
                                from .research import invalid_download
                                raise invalid_download()
                            error_body.extend(chunk)
                        _response(httpx.Response(response.status_code, headers=response.headers,
                                                 content=bytes(error_body)))
                    writer.headers(response)
                    async for chunk in response.aiter_raw(chunk_size=65536):
                        writer.write(chunk)
        except httpx.RequestError:
            raise TransportError("Research download interrupted; no destination was published.",
                                 code="transport_error", retryable=True) from None

    async def datasets(self):
        return (await self.request("GET", "/v1/datasets"))["items"]

    async def predictors(self):
        return (await self.request("GET", "/v1/predictors"))["items"]

    async def jobs(self):
        return (await self.request("GET", "/v1/jobs"))["items"]

    async def usage(self):
        return (await self.request("GET", "/v1/usage"))["items"]

    async def dataset(self, dataset_id):
        return await self.request("GET", f"/v1/datasets/{_id(dataset_id)}")

    async def delete_dataset(self, dataset_id):
        """Delete this owned dataset only if it has no retained/active dependents."""
        return await self.request("DELETE", f"/v1/datasets/{_id(dataset_id)}")

    async def delete_causal_result(self, result_id):
        """Explicitly delete an owned primary graph/scores; retain job/usage metadata."""
        return await self.request("DELETE", f"/v1/causal-results/{_id(result_id)}")

    async def upload(self, *, columns, rows, name="SDK dataset", target=None):
        return await self.request("POST", "/v1/datasets", json=dict(
            columns=columns, rows=rows, name=name, target=target))

    async def submit_fit(self, dataset_id, *, model, task, configuration="default", seed=0,
                         idempotency_key=None):
        result = await self.request("POST", "/v1/fits", json=_fit(
            dataset_id, model, task, configuration, seed),
            headers={"Idempotency-Key": idempotency_key or uuid4().hex})
        return AsyncJob(self, result["id"])

    async def submit_discover(self, dataset_id, *, model="arrow", configuration="default", seed=0,
                              constraints=None, model_version=None, idempotency_key=None):
        payload = _discovery(dataset_id, model, configuration, seed, constraints, model_version)
        result = await self.request("POST", "/v1/discoveries", json=payload,
                                    headers={"Idempotency-Key": idempotency_key or uuid4().hex})
        return AsyncJob(self, result["id"])

    async def discover(self, dataset_id, *, model="arrow", configuration="default", seed=0,
                       constraints=None, model_version=None, idempotency_key=None, timeout=600):
        job = await self.submit_discover(dataset_id, model=model, configuration=configuration,
            seed=seed, constraints=constraints, model_version=model_version, idempotency_key=idempotency_key)
        return await job.result(timeout=timeout)

    async def causal_result(self, job_id, *, _deadline=None):
        from .causal import CausalResult
        from .errors import InvalidCausalResultError
        result = CausalResult(await self.request("GET", f"/v1/jobs/{_id(job_id)}/result", _deadline=_deadline, _job_id=job_id))
        if result.job_id != str(job_id):
            raise InvalidCausalResultError("Causal result belongs to a different discovery job.", code="invalid_causal_result")
        return result

    async def causal_scores(self, job_id):
        from .causal import validated_scores
        result = await self.causal_result(job_id)
        return validated_scores(result, await self.request("GET", f"/v1/jobs/{_id(job_id)}/result/scores"))

    def job(self, job_id):
        return AsyncJob(self, job_id)

    async def predictor(self, predictor_id, *, _deadline=None, _job_id=None):
        return await self.request("GET", f"/v1/predictors/{_id(predictor_id)}", _deadline=_deadline, _job_id=_job_id)

    async def predict(self, predictor_id, *, columns, rows, probabilities=False):
        return await self.request("POST", f"/v1/predictors/{_id(predictor_id)}/predict",
                                  json=dict(columns=columns, rows=rows, probabilities=probabilities))


class AsyncJob:
    def __init__(self, client, job_id):
        value = secret(client.api_key)
        if value and value in str(job_id):
            raise WeltError("Cannot use a job identifier containing the configured credential. "
                            "Inspect jobs before repeating a submission.", code="invalid_response")
        self.client, self.id = client, job_id

    async def inspect(self):
        return await self.client.request("GET", f"/v1/jobs/{_id(self.id)}")

    async def cancel(self):
        return await self.client.request("POST", f"/v1/jobs/{_id(self.id)}/cancel")

    async def result(self, *, timeout=600, poll_interval=0.25):
        """Bound local polling/read awaits; timeout never cancels the server job."""
        deadline = _deadline(timeout, poll_interval)
        while True:
            job = await self.client.request("GET", f"/v1/jobs/{_id(self.id)}", _deadline=deadline, _job_id=self.id)
            if job["status"] in ("succeeded", "partially_completed"):
                if job["status"] == "partially_completed" and job["operation"] != "batch_predict":
                    raise WeltError("Unexpected partial result for this job type.", code="invalid_response")
                if job["operation"] == "discover":
                    return await self.client.causal_result(self.id, _deadline=deadline)
                return (await self.client.request("GET", f"/v1/jobs/{_id(self.id)}/result", _deadline=deadline, _job_id=self.id)
                        if job["operation"] in ("predict", "batch_predict") else await self.client.predictor(job["predictor_id"], _deadline=deadline, _job_id=self.id))
            _failed(job, secret(self.client.api_key))
            await asyncio.sleep(_wait(deadline, self.id, poll_interval))
            poll_interval = min(poll_interval * 1.4, 5)
