"""HTTP resources and durable job handles (API-035)."""

import os
import time
from uuid import uuid4

import httpx

from .errors import (
    AuthenticationError,
    CapacityError,
    ExecutionError,
    InvalidInputError,
    ModelUnavailableError,
    PredictionPendingError,
    WeltError,
)


class Client:
    def __init__(self, base_url=None, api_key=None, *, timeout=60, transport=None):
        self.base_url = base_url or os.getenv("WELT_BASE_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("WELT_API_KEY")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.http = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=timeout, transport=transport
        )

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def request(self, method, path, **kwargs):
        response = self.http.request(method, path, **kwargs)
        if response.is_error:
            try:
                error = response.json()["error"]
            except (ValueError, KeyError, TypeError):
                raise WeltError(f"HTTP {response.status_code} from Welt API.") from None
            code = error.get("code", "unknown")
            cls = WeltError
            if code.startswith("authentication"):
                cls = AuthenticationError
            elif code in (
                "invalid_input",
                "schema_mismatch",
                "workload_limit",
                "payload_limit",
                "unsupported_capability",
            ):
                cls = InvalidInputError
            elif code in ("model_unavailable", "version_unavailable", "worker_unavailable"):
                cls = ModelUnavailableError
            elif code == "capacity_limit":
                cls = CapacityError
            elif code == "execution_failed":
                cls = ExecutionError
            elif code == "prediction_timeout":
                raise PredictionPendingError(
                    error.get("message", code),
                    code=code,
                    request_id=error.get("request_id"),
                    retryable=True,
                    job_id=error.get("job_id"),
                )
            raise cls(
                error.get("message", code),
                code=code,
                request_id=error.get("request_id"),
                retryable=error.get("retryable", False),
            )
        return response.json()

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

    def upload(self, *, columns, rows, name="SDK dataset", target=None):
        return self.request(
            "POST", "/v1/datasets", json=dict(columns=columns, rows=rows, name=name, target=target)
        )

    def submit_fit(
        self, dataset_id, *, model, task, configuration="default", seed=0, idempotency_key=None
    ):
        result = self.request(
            "POST",
            "/v1/fits",
            json=dict(
                dataset_id=dataset_id,
                model_id=model,
                task=task,
                configuration=configuration,
                seed=seed,
            ),
            headers={"Idempotency-Key": idempotency_key or uuid4().hex},
        )
        return Job(self, result["id"])

    def job(self, job_id):
        return Job(self, job_id)

    def predictor(self, predictor_id):
        return self.request("GET", f"/v1/predictors/{predictor_id}")


class Job:
    def __init__(self, client, job_id):
        self.client, self.id = client, job_id

    def inspect(self):
        return self.client.request("GET", f"/v1/jobs/{self.id}")

    def cancel(self):
        return self.client.request("POST", f"/v1/jobs/{self.id}/cancel")

    def result(self, *, timeout=600, poll_interval=0.25):
        deadline = time.monotonic() + timeout
        while True:
            job = self.inspect()
            if job["status"] == "succeeded":
                if job["operation"] == "predict":
                    return self.client.request("GET", f"/v1/jobs/{self.id}/result")
                return self.client.predictor(job["predictor_id"])
            if job["status"] in ("failed", "cancelled"):
                error = job.get("error") or {
                    "code": "job_cancelled",
                    "message": "Job was cancelled.",
                }
                raise ExecutionError(error["message"], code=error["code"])
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Waiting for job {self.id} timed out; the server job continues. Reconnect with Client.job()."
                )
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.4, 5)
