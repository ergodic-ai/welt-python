"""Equivalent synchronous/asynchronous resources and durable job handles."""
import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
    ResultExpiredError, ResultDeletedError, TransportError, WeltError)


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


def _response(response):
    if response.is_error:
        try:
            error = response.json()["error"]
            if not isinstance(error, dict):
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise WeltError(f"HTTP {response.status_code} from Welt API.",
                            status_code=response.status_code) from None
        code = error.get("code", "unknown")
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
        raise cls(error.get("message", "Welt request failed."), code=code,
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


class _HTTP:
    def _configure(self, base_url, api_key, max_retries, max_retry_delay):
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be an integer from 0 to 5.")
        if not math.isfinite(max_retry_delay) or not 0 <= max_retry_delay <= 60:
            raise ValueError("max_retry_delay must be from 0 to 60 seconds.")
        self.base_url = base_url or os.getenv("WELT_BASE_URL", "http://localhost:8080")
        self.api_key = credential(secret(api_key) or os.getenv("WELT_API_KEY"))
        self.max_retries, self.max_retry_delay = max_retries, max_retry_delay
        value = secret(self.api_key)
        return {"Authorization": f"Bearer {value}"} if value else {}

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
                 max_retries=2, max_retry_delay=5):
        headers = self._configure(base_url, api_key, max_retries, max_retry_delay)
        self.http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout,
                                 transport=transport, follow_redirects=False)

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def request(self, method, path, **kwargs):
        for attempt in range(self.max_retries + 1):
            try:
                response = self.http.request(method, path, **kwargs)
            except httpx.RequestError:
                delay = self._delay(method, attempt)
                if delay is None:
                    raise TransportError("Could not reach Welt API; reconnect to accepted jobs.",
                                         code="transport_error", retryable=True) from None
            else:
                delay = self._delay(method, attempt, response)
                if delay is None:
                    return _response(response)
            time.sleep(delay)

    def models(self):
        return self.request("GET", "/v1/models")["items"]

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

    def upload_info(self, upload_id):
        return self.request("GET", f"/v1/uploads/{_id(upload_id)}")

    def upload_file(self, path, *, target=None, schema=None):
        """Upload bounded binary chunks; failures expose upload_id for resume."""
        from .uploads import declaration
        declared = declaration(path, target, schema)
        info = self.request("POST", "/v1/uploads", json=declared)
        return self._send_file(info, path, declared)

    def resume_upload(self, upload_id, path, *, target=None, schema=None):
        from .uploads import declaration
        return self._send_file(self.upload_info(upload_id), path, declaration(path, target, schema), upload_id)

    def _send_file(self, info, path, declared, upload_id=None):
        from .uploads import annotate, chunks, matches
        try:
            matches(info, declared, upload_id)
            for index, body, digest in chunks(path, info, declared):
                self.request("PUT", f"/v1/uploads/{_id(info['id'])}/chunks/{index}", content=body,
                    headers={"Content-Type": "application/octet-stream", "X-Chunk-SHA256": digest})
            return self.request("POST", f"/v1/uploads/{_id(info['id'])}/complete")
        except (WeltError, OSError, asyncio.CancelledError) as error:
            known_id = upload_id or (info.get("id") if isinstance(info, dict) else None)
            if isinstance(known_id, str) and known_id:
                annotate(error, known_id)
            raise

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

    def causal_result(self, job_id):
        from .causal import CausalResult
        from .errors import InvalidCausalResultError
        result = CausalResult(self.request("GET", f"/v1/jobs/{_id(job_id)}/result"))
        if result.job_id != str(job_id):
            raise InvalidCausalResultError("Causal result belongs to a different discovery job.", code="invalid_causal_result")
        return result

    def causal_scores(self, job_id):
        from .causal import validated_scores
        result = self.causal_result(job_id)
        return validated_scores(result, self.request("GET", f"/v1/jobs/{_id(job_id)}/result/scores"))

    def job(self, job_id):
        return Job(self, job_id)

    def predictor(self, predictor_id):
        return self.request("GET", f"/v1/predictors/{_id(predictor_id)}")

    def predict(self, predictor_id, *, columns, rows, probabilities=False):
        return self.request("POST", f"/v1/predictors/{_id(predictor_id)}/predict",
                            json=dict(columns=columns, rows=rows, probabilities=probabilities))


def _failed(job):
    if job["status"] in ("failed", "cancelled"):
        error = job.get("error") or {"code": "job_cancelled", "message": "Job was cancelled."}
        cls = JobCancelledError if job["status"] == "cancelled" else ExecutionError
        raise cls(error["message"], code=error["code"], job_id=job["id"],
                  request_id=error.get("request_id"), retryable=error.get("retryable", False))


def _deadline(timeout, poll_interval):
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout must be finite and nonnegative.")
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("poll_interval must be finite and positive.")
    return time.monotonic() + timeout


def _wait(deadline, job_id, interval):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise JobTimeoutError("Waiting timed out; the server job continues. Reconnect with client.job().",
                              code="job_wait_timeout", job_id=job_id, retryable=True)
    return min(interval, remaining)


class Job:
    def __init__(self, client, job_id):
        self.client, self.id = client, job_id

    def inspect(self):
        return self.client.request("GET", f"/v1/jobs/{_id(self.id)}")

    def cancel(self):
        return self.client.request("POST", f"/v1/jobs/{_id(self.id)}/cancel")

    def result(self, *, timeout=600, poll_interval=0.25):
        deadline = _deadline(timeout, poll_interval)
        while True:
            job = self.inspect()
            if job["status"] == "succeeded":
                if job["operation"] == "discover":
                    return self.client.causal_result(self.id)
                return (self.client.request("GET", f"/v1/jobs/{_id(self.id)}/result")
                        if job["operation"] == "predict" else self.client.predictor(job["predictor_id"]))
            _failed(job)
            time.sleep(_wait(deadline, self.id, poll_interval))
            poll_interval = min(poll_interval * 1.4, 5)


class AsyncClient(_HTTP):
    def __init__(self, base_url=None, api_key=None, *, timeout=60, transport=None,
                 max_retries=2, max_retry_delay=5):
        headers = self._configure(base_url, api_key, max_retries, max_retry_delay)
        self.http = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout,
                                      transport=transport, follow_redirects=False)

    async def aclose(self):
        await self.http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def request(self, method, path, **kwargs):
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.http.request(method, path, **kwargs)
            except httpx.RequestError:
                delay = self._delay(method, attempt)
                if delay is None:
                    raise TransportError("Could not reach Welt API; reconnect to accepted jobs.",
                                         code="transport_error", retryable=True) from None
            else:
                delay = self._delay(method, attempt, response)
                if delay is None:
                    return _response(response)
            await asyncio.sleep(delay)

    async def models(self):
        return (await self.request("GET", "/v1/models"))["items"]

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

    async def upload_info(self, upload_id):
        return await self.request("GET", f"/v1/uploads/{_id(upload_id)}")

    async def upload_file(self, path, *, target=None, schema=None):
        from .uploads import declaration
        declared = await asyncio.to_thread(declaration, path, target, schema)
        info = await self.request("POST", "/v1/uploads", json=declared)
        return await self._send_file(info, path, declared)

    async def resume_upload(self, upload_id, path, *, target=None, schema=None):
        from .uploads import declaration
        declared = await asyncio.to_thread(declaration, path, target, schema)
        return await self._send_file(await self.upload_info(upload_id), path, declared, upload_id)

    async def _send_file(self, info, path, declared, upload_id=None):
        from .uploads import annotate, chunks, matches
        try:
            matches(info, declared, upload_id)
            iterator = chunks(path, info, declared)
            # Avoid synchronous disk reads on the async HTTP event loop.
            def advance():
                return next(iterator, None)
            async def next_chunk():
                task = asyncio.create_task(asyncio.to_thread(advance))
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError as cancelled:
                    # Finish this bounded read before closing its generator.
                    while not task.done():
                        try:
                            await asyncio.shield(task)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    try:
                        task.result()
                    except (Exception, asyncio.CancelledError):
                        pass
                    raise cancelled
            try:
                while item := await next_chunk():
                    index, body, digest = item
                    await self.request("PUT", f"/v1/uploads/{_id(info['id'])}/chunks/{index}", content=body,
                        headers={"Content-Type": "application/octet-stream", "X-Chunk-SHA256": digest})
            finally:
                iterator.close()
            return await self.request("POST", f"/v1/uploads/{_id(info['id'])}/complete")
        except (WeltError, OSError, asyncio.CancelledError) as error:
            known_id = upload_id or (info.get("id") if isinstance(info, dict) else None)
            if isinstance(known_id, str) and known_id:
                annotate(error, known_id)
            raise

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

    async def causal_result(self, job_id):
        from .causal import CausalResult
        from .errors import InvalidCausalResultError
        result = CausalResult(await self.request("GET", f"/v1/jobs/{_id(job_id)}/result"))
        if result.job_id != str(job_id):
            raise InvalidCausalResultError("Causal result belongs to a different discovery job.", code="invalid_causal_result")
        return result

    async def causal_scores(self, job_id):
        from .causal import validated_scores
        result = await self.causal_result(job_id)
        return validated_scores(result, await self.request("GET", f"/v1/jobs/{_id(job_id)}/result/scores"))

    def job(self, job_id):
        return AsyncJob(self, job_id)

    async def predictor(self, predictor_id):
        return await self.request("GET", f"/v1/predictors/{_id(predictor_id)}")

    async def predict(self, predictor_id, *, columns, rows, probabilities=False):
        return await self.request("POST", f"/v1/predictors/{_id(predictor_id)}/predict",
                                  json=dict(columns=columns, rows=rows, probabilities=probabilities))


class AsyncJob:
    def __init__(self, client, job_id):
        self.client, self.id = client, job_id

    async def inspect(self):
        return await self.client.request("GET", f"/v1/jobs/{_id(self.id)}")

    async def cancel(self):
        return await self.client.request("POST", f"/v1/jobs/{_id(self.id)}/cancel")

    async def result(self, *, timeout=600, poll_interval=0.25):
        deadline = _deadline(timeout, poll_interval)
        while True:
            job = await self.inspect()
            if job["status"] == "succeeded":
                if job["operation"] == "discover":
                    return await self.client.causal_result(self.id)
                return (await self.client.request("GET", f"/v1/jobs/{_id(self.id)}/result")
                        if job["operation"] == "predict" else await self.client.predictor(job["predictor_id"]))
            _failed(job)
            await asyncio.sleep(_wait(deadline, self.id, poll_interval))
            poll_interval = min(poll_interval * 1.4, 5)
