"""Stable remote errors with actionable metadata."""

import re


def safe_id(value):
    """Only bounded opaque identifiers are suitable for diagnostic display."""
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) else None


class WeltError(Exception):
    def __init__(self, message, *, code="unknown", request_id=None, retryable=False,
                 job_id=None, status_code=None, retry_after=None):
        self.message = str(message)
        request_id, job_id = safe_id(request_id), safe_id(job_id)
        super().__init__(self.message)
        self.code, self.request_id, self.retryable = code, request_id, retryable
        self.job_id, self.status_code, self.retry_after = job_id, status_code, retry_after


    def __str__(self):
        context = []
        if self.request_id:
            context.append(f"request {self.request_id}")
        if self.job_id:
            context.append(f"job {self.job_id}")
        return self.message + (" [" + "; ".join(context) + "]" if context else "")


class AuthenticationError(WeltError):
    pass


class InvalidInputError(WeltError, ValueError):
    pass


class ModelUnavailableError(WeltError):
    pass


class CapacityError(WeltError):
    pass


class ExecutionError(WeltError):
    pass


class PermissionDeniedError(WeltError):
    pass


class NotFoundError(WeltError):
    pass


class ConflictError(WeltError):
    pass


class RateLimitError(CapacityError):
    pass


class ResultExpiredError(WeltError):
    pass


class ResultDeletedError(WeltError):
    pass


class TransportError(WeltError):
    pass


class JobTimeoutError(WeltError, TimeoutError):
    pass


class JobCancelledError(ExecutionError):
    pass


class PredictionPendingError(WeltError):
    def __init__(self, message, *, job_id=None, **kwargs):
        super().__init__(message, job_id=job_id, **kwargs)


class InvalidCausalResultError(WeltError, ValueError):
    pass


class UnsupportedGraphConversionError(WeltError, ValueError):
    pass


class OptionalDependencyError(WeltError, ImportError):
    pass
