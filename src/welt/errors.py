"""Stable remote errors with actionable metadata."""


class WeltError(Exception):
    def __init__(self, message, *, code="unknown", request_id=None, retryable=False,
                 job_id=None, status_code=None, retry_after=None):
        super().__init__(message)
        self.code, self.request_id, self.retryable = code, request_id, retryable
        self.job_id, self.status_code, self.retry_after = job_id, status_code, retry_after


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


class InvalidBatchResultError(WeltError, ValueError):
    pass
